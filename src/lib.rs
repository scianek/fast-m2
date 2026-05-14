//! Rust core for fast-m2: M2 scorer for grammatical error correction evaluation.
//!
//! Implements the max-match scoring algorithm from Dahlmeier & Ng (2012). Given a
//! system hypothesis and a set of gold annotations in M2 format, the scorer extracts
//! edit sequences from both, aligns them via a weighted edit graph, and computes
//! corpus-level precision, recall, and F_beta.
//!
//! The main entry point exposed to Python is [`batch_multi_pre_rec_f1`], which takes
//! parallel lists of hypothesis sentences and gold annotation maps and returns the
//! three metrics as a tuple. All other functions are internal pipeline stages.

use pyo3::prelude::*;
use std::collections::HashMap;

/// A single edit operation on a token span, extracted from a Levenshtein alignment.
///
/// `start` and `end` are token offsets into the source sentence (end is exclusive).
/// `orig` is the source token(s) in that span, `corr` is the replacement string.
/// `unchanged_words` counts how many tokens passed through unmodified; this is
/// used by `transitive_arcs` to enforce the `max_unchanged_words` cap on merged spans.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Edit {
    kind: EditKind,
    start: usize,
    end: usize,
    orig: String,
    corr: String,
    unchanged_words: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum EditKind {
    Ins,
    Del,
    Sub,
    Noop,
}

/// A single annotation from an M2 file, representing one acceptable correction.
///
/// `start` and `end` are token offsets (i64 to accommodate the -1 noop sentinel).
/// `corrections` holds every acceptable correction string for this span; a system
/// edit is considered correct if its `corr` appears anywhere in this list.
#[derive(Clone, Debug)]
struct GoldEdit {
    start: i64,
    end: i64,
    orig: String,
    corrections: Vec<String>,
}

/// A cell position (row, col) in the Levenshtein matrix, used as a graph vertex.
type Vertex = (usize, usize);

/// A directed edge between two vertices in the edit graph.
type Edge = (Vertex, Vertex);

/// A directed acyclic graph of edit operations derived from a Levenshtein matrix.
///
/// Each edge corresponds to one edit (ins/del/sub/noop) and carries a distance
/// weight. Weights start at 1.0 and are adjusted by `set_weights` to reward
/// edges that match gold edits (negative weight) and penalise spurious ones
/// (small positive epsilon), so that Bellman-Ford finds the best-matching path.
struct EditGraph {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    dist: HashMap<Edge, f64>,
    edits: HashMap<Edge, Edit>,
}

/// Computes the Levenshtein distance matrix and a backpointer table between
/// `first` (source tokens) and `second` (hypothesis tokens).
///
/// Returns the filled distance matrix and a map from every reachable vertex to
/// its predecessor vertices, along with the edit that produced each transition.
/// A vertex can have multiple backpointers when several operations tie on cost,
/// which is why the graph can branch. `cost_ins`, `cost_del`, and `cost_sub`
/// are configurable; the scorer calls this function twice with sub-cost 1 and 2
/// to capture alternative alignments, then merges the resulting graphs.
fn levenshtein_matrix(
    first: &[String],
    second: &[String],
    cost_ins: u32,
    cost_del: u32,
    cost_sub: u32,
) -> (Vec<Vec<u32>>, HashMap<Vertex, Vec<(Vertex, Edit)>>) {
    let n = first.len() + 1;
    let m = second.len() + 1;

    let mut mat = vec![vec![0u32; m]; n];
    let mut backpointers: HashMap<Vertex, Vec<(Vertex, Edit)>> = HashMap::new();

    for i in 1..n {
        mat[i][0] = i as u32;
        let edit = Edit {
            kind: EditKind::Del,
            start: i - 1,
            end: i,
            orig: first[i - 1].clone(),
            corr: String::new(),
            unchanged_words: 0,
        };
        backpointers
            .entry((i, 0))
            .or_default()
            .push(((i - 1, 0), edit));
    }

    for j in 1..m {
        mat[0][j] = j as u32;
        let edit = Edit {
            kind: EditKind::Ins,
            start: 0,
            end: 0, // zero-width span: insertion adds tokens without consuming source
            orig: String::new(),
            corr: second[j - 1].clone(),
            unchanged_words: 0,
        };
        backpointers
            .entry((0, j))
            .or_default()
            .push(((0, j - 1), edit));
    }

    for i in 1..n {
        for j in 1..m {
            let same = first[i - 1] == second[j - 1];
            let sub_cost = if same { 0 } else { cost_sub };
            let sub_val = mat[i - 1][j - 1] + sub_cost;
            let del_val = mat[i - 1][j] + cost_del;
            let ins_val = mat[i][j - 1] + cost_ins;
            let best = sub_val.min(del_val).min(ins_val);
            mat[i][j] = best;

            // All operations that tie on cost are recorded as backpointers so the
            // graph captures every optimal alignment, not just one.
            if sub_val == best {
                let edit = Edit {
                    kind: if same { EditKind::Noop } else { EditKind::Sub },
                    start: i - 1,
                    end: i,
                    orig: first[i - 1].clone(),
                    corr: second[j - 1].clone(),
                    unchanged_words: if same { 1 } else { 0 },
                };
                backpointers
                    .entry((i, j))
                    .or_default()
                    .push(((i - 1, j - 1), edit));
            }
            if del_val == best {
                let edit = Edit {
                    kind: EditKind::Del,
                    start: i - 1,
                    end: i,
                    orig: first[i - 1].clone(),
                    corr: String::new(),
                    unchanged_words: 0,
                };
                backpointers
                    .entry((i, j))
                    .or_default()
                    .push(((i - 1, j), edit));
            }
            if ins_val == best {
                let edit = Edit {
                    kind: EditKind::Ins,
                    start: i,
                    end: i, // zero-width span at source position i
                    orig: String::new(),
                    corr: second[j - 1].clone(),
                    unchanged_words: 0,
                };
                backpointers
                    .entry((i, j))
                    .or_default()
                    .push(((i, j - 1), edit));
            }
        }
    }

    (mat, backpointers)
}

/// Constructs an `EditGraph` from a Levenshtein matrix and its backpointer table.
///
/// Performs a BFS from the bottom-right corner of the matrix (the vertex
/// representing the complete source-hypothesis alignment) backwards through all
/// predecessor vertices. Every transition becomes a directed edge in the graph,
/// initialised with distance 1.0. The resulting graph is a DAG from (0,0) to (n,m).
fn build_edit_graph(
    mat: &[Vec<u32>],
    backpointers: &HashMap<Vertex, Vec<(Vertex, Edit)>>,
) -> EditGraph {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut edges: Vec<Edge> = Vec::new();
    let mut dist: HashMap<Edge, f64> = HashMap::new();
    let mut edits: HashMap<Edge, Edit> = HashMap::new();

    let v_start: Vertex = (mat.len() - 1, mat[0].len() - 1);
    let mut queue: Vec<Vertex> = vec![v_start];
    let mut visited: std::collections::HashSet<Vertex> = std::collections::HashSet::new();

    while !queue.is_empty() {
        let v = queue.remove(0);
        if visited.contains(&v) {
            continue;
        }
        visited.insert(v);
        vertices.push(v);

        if let Some(preds) = backpointers.get(&v) {
            for (vprev, edit) in preds {
                let edge: Edge = (*vprev, v);
                edges.push(edge);
                dist.insert(edge, 1.0);
                edits.insert(edge, edit.clone());
                if !visited.contains(vprev) {
                    queue.push(*vprev);
                }
            }
        }
    }

    EditGraph {
        vertices,
        edges,
        dist,
        edits,
    }
}

/// Merges two `EditGraph`s into one by taking the union of their vertices, edges,
/// distances, and edit maps.
///
/// When the same edge appears in both graphs, the entry from `g1` is kept for
/// both distance and edit content. This is used to combine the two graphs built
/// from the sub-cost-1 and sub-cost-2 Levenshtein matrices, giving the scorer
/// access to a richer set of candidate alignments.
fn merge_graphs(g1: EditGraph, g2: EditGraph) -> EditGraph {
    let mut vertices = g1.vertices;
    for v in g2.vertices {
        if !vertices.contains(&v) {
            vertices.push(v);
        }
    }
    vertices.sort();

    let mut edges = g1.edges;
    for e in g2.edges {
        if !edges.contains(&e) {
            edges.push(e);
        }
    }
    edges.sort();

    let mut dist = g1.dist;
    for (k, v) in g2.dist {
        dist.entry(k).or_insert(v);
    }

    let mut edits = g1.edits;
    for (e, ed) in g2.edits {
        edits.entry(e).or_insert(ed);
    }

    EditGraph {
        vertices,
        edges,
        dist,
        edits,
    }
}

/// Combines two adjacent edits into a single edit spanning both their token ranges.
///
/// The resulting `EditKind` is determined by the combination of input kinds: for
/// example, Del+Ins becomes Sub, Noop+Noop stays Noop, and anything involving a
/// non-noop operation produces Sub or preserves the dominant operation type.
/// `unchanged_words` is summed so callers can gate on `max_unchanged_words`.
fn merge_edits(e1: &Edit, e2: &Edit) -> Edit {
    let join = |a: &str, b: &str| format!("{} {}", a, b);
    let (kind, start, end, orig, corr) = match (&e1.kind, &e2.kind) {
        (EditKind::Ins, EditKind::Ins) => (
            EditKind::Ins,
            e1.start,
            e2.end,
            String::new(),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Ins, EditKind::Del) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            e2.orig.clone(),
            e1.corr.clone(),
        ),
        (EditKind::Ins, EditKind::Sub) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            e2.orig.clone(),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Ins, EditKind::Noop) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            e2.orig.clone(),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Del, EditKind::Ins) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            e1.orig.clone(),
            e2.corr.clone(),
        ),
        (EditKind::Del, EditKind::Del) => (
            EditKind::Del,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            String::new(),
        ),
        (EditKind::Del, EditKind::Sub) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            e2.corr.clone(),
        ),
        (EditKind::Del, EditKind::Noop) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            e2.corr.clone(),
        ),
        (EditKind::Sub, EditKind::Ins) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            e1.orig.clone(),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Sub, EditKind::Del) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            e1.corr.clone(),
        ),
        (EditKind::Sub, EditKind::Sub) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Sub, EditKind::Noop) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Noop, EditKind::Ins) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            e1.orig.clone(),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Noop, EditKind::Del) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            e1.corr.clone(),
        ),
        (EditKind::Noop, EditKind::Sub) => (
            EditKind::Sub,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            join(&e1.corr, &e2.corr),
        ),
        (EditKind::Noop, EditKind::Noop) => (
            EditKind::Noop,
            e1.start,
            e2.end,
            join(&e1.orig, &e2.orig),
            join(&e1.corr, &e2.corr),
        ),
    };
    Edit {
        kind,
        start,
        end,
        orig,
        corr,
        unchanged_words: e1.unchanged_words + e2.unchanged_words,
    }
}

/// Extends the graph with transitive arcs that represent multi-token edits.
///
/// Uses a Floyd-Warshall-style triple loop: for every pair of adjacent edges
/// (vi→vk) and (vk→vj), if their combined distance is shorter than the current
/// vi→vj distance, a new arc is added by merging the two edits via `merge_edits`.
/// An arc is only added when the merged edit's `unchanged_words` count stays within
/// `max_unchanged_words`, which controls how wide a span the scorer considers.
///
/// After all arcs are added, transitive noop arcs (distance > 1.0) are removed
/// because they span multiple unchanged tokens and do not represent real edits.
fn transitive_arcs(mut graph: EditGraph, max_unchanged_words: usize) -> EditGraph {
    let v = graph.vertices.clone();
    let n = v.len();

    for k in 0..n {
        let vk = v[k];
        for i in 0..n {
            let vi = v[i];
            let eik = match graph.edits.get(&(vi, vk)) {
                Some(e) => e.clone(),
                None => continue,
            };
            let dik = *graph.dist.get(&(vi, vk)).unwrap_or(&f64::INFINITY);
            for j in 0..n {
                let vj = v[j];
                let ekj = match graph.edits.get(&(vk, vj)) {
                    Some(e) => e.clone(),
                    None => continue,
                };
                let dkj = *graph.dist.get(&(vk, vj)).unwrap_or(&f64::INFINITY);
                let cur = *graph.dist.get(&(vi, vj)).unwrap_or(&f64::INFINITY);
                if dik + dkj < cur {
                    let merged = merge_edits(&eik, &ekj);
                    if merged.unchanged_words <= max_unchanged_words {
                        graph.edges.push((vi, vj));
                        graph.dist.insert((vi, vj), dik + dkj);
                        graph.edits.insert((vi, vj), merged);
                    }
                }
            }
        }
    }

    let to_remove: Vec<Edge> = graph
        .edges
        .iter()
        .filter(|&&e| {
            graph
                .edits
                .get(&e)
                .map_or(false, |ed| ed.kind == EditKind::Noop)
                && *graph.dist.get(&e).unwrap_or(&0.0) > 1.0
        })
        .cloned()
        .collect();

    for e in to_remove {
        graph.edges.retain(|x| x != &e);
        graph.dist.insert(e, f64::INFINITY);
        graph.edits.remove(&e);
    }

    graph
}

/// Returns true if a system edit matches a gold edit.
///
/// A match requires identical span (start, end), identical original text, and
/// the system's correction appearing in the gold's accepted corrections list.
fn edit_matches_gold(edit: &Edit, gold: &GoldEdit) -> bool {
    edit.start == gold.start as usize
        && edit.end == gold.end as usize
        && edit.orig == gold.orig
        && gold.corrections.contains(&edit.corr)
}

/// Assigns weights to graph edges based on how well they match the gold edits.
///
/// Edges whose edit matches a gold edit receive a large negative weight (-|E|),
/// making them strongly preferred by Bellman-Ford. Non-matching, non-noop edges
/// receive a small positive epsilon to break ties against noop paths. Noop edges
/// are left unchanged.
///
/// Insertion edges (where start == end, i.e. zero-width spans) require special
/// handling: multiple insertions can occur at the same position, so a bidirectional
/// pointer scheme is used to greedily match them left-to-right and right-to-left
/// alternately, ensuring each gold edit is claimed by at most one system edit.
/// Deletion and substitution edges are simpler - any edge matching any gold edit
/// at the same span receives the reward, independently of order.
fn set_weights(graph: &EditGraph, gold_edits: &[GoldEdit]) -> HashMap<Edge, f64> {
    const EPSILON: f64 = 0.001;
    let num_edges = graph.edges.len() as f64;

    let mut ret_dist = graph.dist.clone();

    let mut m: HashMap<(usize, usize), Vec<Edge>> = HashMap::new();
    for &edge in &graph.edges {
        if let Some(ed) = graph.edits.get(&edge) {
            m.entry((ed.start, ed.end)).or_default().push(edge);
        }
    }
    for edges in m.values_mut() {
        edges.sort();
    }

    let mut g: HashMap<(usize, usize), Vec<&GoldEdit>> = HashMap::new();
    for gold in gold_edits {
        if gold.start >= 0 && gold.end >= 0 {
            g.entry((gold.start as usize, gold.end as usize))
                .or_default()
                .push(gold);
        }
    }

    for (&span, span_edges) in &m {
        let empty: Vec<&GoldEdit> = Vec::new();
        let gold_list = g.get(&span).unwrap_or(&empty);

        if span.0 == span.1 {
            if span_edges.is_empty() {
                continue;
            }
            let mut lptr = 0usize;
            let mut rptr = span_edges.len() - 1;
            let mut cur = lptr;
            let mut g_lptr = 0usize;
            let mut g_rptr = gold_list.len().saturating_sub(1);

            while lptr <= rptr {
                let edge = span_edges[cur];
                let this_edit = match graph.edits.get(&edge) {
                    Some(e) => e,
                    None => break,
                };

                // Alternate search direction based on which pointer is active.
                let cur_gold_range: Vec<usize> = if cur == lptr {
                    (g_lptr..=g_rptr.min(gold_list.len().saturating_sub(1))).collect()
                } else {
                    (g_lptr..=g_rptr.min(gold_list.len().saturating_sub(1)))
                        .rev()
                        .collect()
                };

                let mut has_match = false;
                let mut matched_i = 0;
                for &i in &cur_gold_range {
                    if i < gold_list.len() && edit_matches_gold(this_edit, gold_list[i]) {
                        has_match = true;
                        matched_i = i;
                        ret_dist.insert(edge, -num_edges);
                        if cur == lptr {
                            g_lptr = i + 1;
                        } else if i > 0 {
                            g_rptr = i - 1;
                        } else {
                            break;
                        }
                        break;
                    }
                }
                let _ = matched_i;

                if !has_match && this_edit.kind != EditKind::Noop {
                    *ret_dist.entry(edge).or_insert(1.0) += EPSILON;
                }

                if has_match {
                    if cur == lptr {
                        lptr += 1;
                        while lptr < span_edges.len() && span_edges[lptr].0 != span_edges[cur].1 {
                            let e2 = span_edges[lptr];
                            if graph
                                .edits
                                .get(&e2)
                                .map_or(false, |ed| ed.kind != EditKind::Noop)
                            {
                                *ret_dist.entry(e2).or_insert(1.0) += EPSILON;
                            }
                            lptr += 1;
                        }
                        cur = lptr;
                    } else {
                        if rptr > 0 {
                            rptr -= 1;
                        } else {
                            break;
                        }
                        while rptr > 0 && span_edges[rptr].1 != span_edges[cur].0 {
                            let e2 = span_edges[rptr];
                            if graph
                                .edits
                                .get(&e2)
                                .map_or(false, |ed| ed.kind != EditKind::Noop)
                            {
                                *ret_dist.entry(e2).or_insert(1.0) += EPSILON;
                            }
                            rptr -= 1;
                        }
                        cur = rptr;
                    }
                } else if cur == lptr {
                    lptr += 1;
                    cur = rptr;
                } else {
                    if rptr > 0 {
                        rptr -= 1;
                    }
                    cur = lptr;
                }
            }
        } else {
            for &edge in span_edges {
                let this_edit = match graph.edits.get(&edge) {
                    Some(e) => e,
                    None => continue,
                };
                let mut has_match = false;
                for gold in gold_list {
                    if edit_matches_gold(this_edit, gold) {
                        has_match = true;
                        ret_dist.insert(edge, -num_edges);
                        break;
                    }
                }
                if !has_match && this_edit.kind != EditKind::Noop {
                    *ret_dist.entry(edge).or_insert(1.0) += EPSILON;
                }
            }
        }
    }

    ret_dist
}

/// Finds the edit sequence through the graph that matches the most gold edits,
/// using Bellman-Ford shortest-path on the weighted graph.
///
/// Because gold-matching edges have negative weight and spurious edges carry a
/// small positive epsilon, the minimum-cost path is also the maximally
/// gold-matching path. The path is recovered by backtracking from the last
/// (bottom-right) vertex. Noop edges are excluded from the returned sequence
/// since they represent unchanged tokens, not proposed corrections.
fn best_edit_seq_bf(graph: &EditGraph, dist: &HashMap<Edge, f64>) -> Vec<Edit> {
    let mut this_dist: HashMap<Vertex, f64> = HashMap::new();
    let mut path: HashMap<Vertex, Vertex> = HashMap::new();

    for &v in &graph.vertices {
        this_dist.insert(v, f64::INFINITY);
    }
    this_dist.insert((0, 0), 0.0);

    for _ in 0..graph.vertices.len().saturating_sub(1) {
        for &(vi, vw) in &graph.edges {
            let d_edge = *dist.get(&(vi, vw)).unwrap_or(&f64::INFINITY);
            let d_vi = *this_dist.get(&vi).unwrap_or(&f64::INFINITY);
            if d_vi + d_edge < *this_dist.get(&vw).unwrap_or(&f64::INFINITY) {
                this_dist.insert(vw, d_vi + d_edge);
                path.insert(vw, vi);
            }
        }
    }

    let mut v = *graph.vertices.iter().max().unwrap_or(&(0, 0));
    let mut edit_seq: Vec<Edit> = Vec::new();

    loop {
        match path.get(&v) {
            None => break,
            Some(&w) => {
                if let Some(ed) = graph.edits.get(&(w, v)).cloned() {
                    if ed.kind != EditKind::Noop {
                        edit_seq.push(ed);
                    }
                }
                v = w;
            }
        }
    }

    edit_seq
}

/// Returns the subset of `edit_seq` that matches entries in `gold_edits`.
///
/// Iterates the edit sequence in source order (the sequence comes out of
/// Bellman-Ford in reverse, so we iterate reversed) and for each edit performs
/// a linear scan through the gold list starting from where the previous match
/// left off. This preserves ordering: a gold edit can only be claimed once, and
/// only by the first system edit that matches it in left-to-right order.
fn match_seq(edit_seq: &[Edit], gold_edits: &[GoldEdit]) -> Vec<Edit> {
    let mut matched: Vec<Edit> = Vec::new();
    let gold_seq: Vec<&GoldEdit> = gold_edits.iter().collect();
    let mut last_index = 0usize;

    for edit in edit_seq.iter().rev() {
        for i in last_index..gold_seq.len() {
            let g = gold_seq[i];
            if edit.start == g.start as usize
                && edit.end == g.end as usize
                && edit.orig == g.orig
                && g.corrections.contains(&edit.corr)
            {
                matched.push(edit.clone());
                last_index = i + 1;
                break;
            }
        }
    }
    matched
}

/// Returns true if two strings are equal when whitespace and casing are ignored.
fn equals_ignore_whitespace_casing(a: &str, b: &str) -> bool {
    a.replace(' ', "").to_lowercase() == b.replace(' ', "").to_lowercase()
}

/// Scores a single hypothesis sentence against its gold annotations.
///
/// Builds two Levenshtein graphs (sub-cost 1 and 2), merges them, adds
/// transitive arcs, then for each annotator runs `set_weights` and
/// `best_edit_seq_bf` to find the best-matching edit sequence. Returns the
/// (correct, proposed, gold) triple for the annotator that maximises F_beta
/// for this sentence. Annotator selection is local to the sentence rather than
/// cumulative across the corpus, which allows sentences to be scored independently.
fn score_single(
    candidate: &str,
    source: &str,
    golds_set: &HashMap<u32, Vec<GoldEdit>>,
    max_unchanged_words: usize,
    beta: f64,
    ignore_whitespace_casing: bool,
) -> (f64, f64, f64) {
    let candidate_tok: Vec<String> = candidate.split_whitespace().map(String::from).collect();
    let source_tok: Vec<String> = source.split_whitespace().map(String::from).collect();

    let (mat1, bp1) = levenshtein_matrix(&source_tok, &candidate_tok, 1, 1, 1);
    let (mat2, bp2) = levenshtein_matrix(&source_tok, &candidate_tok, 1, 1, 2);

    let g1 = build_edit_graph(&mat1, &bp1);
    let g2 = build_edit_graph(&mat2, &bp2);
    let mut graph = merge_graphs(g1, g2);
    graph = transitive_arcs(graph, max_unchanged_words);

    let sqbeta = beta * beta;
    let mut best_f1 = -1.0f64;
    let mut best_sc = -1.0f64;
    let mut best_sp = f64::INFINITY;
    let mut best_sg = f64::INFINITY;
    let mut argmax_correct = 0.0f64;
    let mut argmax_proposed = 0.0f64;
    let mut argmax_gold = 0.0f64;

    for gold_list in golds_set.values() {
        let local_dist = set_weights(&graph, gold_list);
        let mut edit_seq = best_edit_seq_bf(&graph, &local_dist);

        if ignore_whitespace_casing {
            edit_seq.retain(|e| !equals_ignore_whitespace_casing(&e.orig, &e.corr));
        }

        let correct = match_seq(&edit_seq, gold_list);

        let sc = correct.len() as f64;
        let sp = edit_seq.len() as f64;
        let sg = gold_list.len() as f64;

        let f1 = {
            let denom = sqbeta * sg + sp;
            if denom == 0.0 {
                if sc == 0.0 {
                    1.0
                } else {
                    0.0
                }
            } else {
                (1.0 + sqbeta) * sc / denom
            }
        };

        let is_better = f1 > best_f1
            || (f1 == best_f1 && sc > best_sc)
            || (f1 == best_f1 && sc == best_sc && sp + sqbeta * sg < best_sp + sqbeta * best_sg);

        if is_better {
            best_f1 = f1;
            best_sc = sc;
            best_sp = sp;
            best_sg = sg;
            argmax_correct = sc;
            argmax_proposed = sp;
            argmax_gold = sg;
        }
    }

    (argmax_correct, argmax_proposed, argmax_gold)
}

/// Scores a batch of hypothesis sentences against gold M2 annotations and returns
/// corpus-level precision, recall, and F_beta.
///
/// `candidates` and `sources` are parallel lists of tokenised sentences.
/// `gold_edits_raw` is a list of per-sentence annotation maps: each map has
/// annotator IDs as keys and lists of (start, end, original, corrections) tuples
/// as values, matching the structure produced by the Python M2 parser.
///
/// Each sentence is scored independently via `score_single`. Correct, proposed,
/// and gold edit counts are accumulated across the corpus and used to compute
/// the final metrics. Precision defaults to 1.0 when no edits are proposed;
/// recall defaults to 1.0 when the gold set is empty.
#[pyfunction]
#[pyo3(signature = (
    candidates,
    sources,
    gold_edits_raw,
    max_unchanged_words = 2,
    beta = 0.5,
    ignore_whitespace_casing = false,
))]
fn batch_multi_pre_rec_f1(
    candidates: Vec<String>,
    sources: Vec<String>,
    gold_edits_raw: Vec<HashMap<u32, Vec<(i64, i64, String, Vec<String>)>>>,
    max_unchanged_words: usize,
    beta: f64,
    ignore_whitespace_casing: bool,
) -> PyResult<(f64, f64, f64)> {
    assert_eq!(candidates.len(), sources.len());
    assert_eq!(candidates.len(), gold_edits_raw.len());

    let gold_edits: Vec<HashMap<u32, Vec<GoldEdit>>> = gold_edits_raw
        .into_iter()
        .map(|ann_map| {
            ann_map
                .into_iter()
                .map(|(ann_id, edits)| {
                    let gold = edits
                        .into_iter()
                        .map(|(start, end, orig, corrections)| GoldEdit {
                            start,
                            end,
                            orig,
                            corrections,
                        })
                        .collect();
                    (ann_id, gold)
                })
                .collect()
        })
        .collect();

    let mut stat_correct = 0.0f64;
    let mut stat_proposed = 0.0f64;
    let mut stat_gold = 0.0f64;

    for ((candidate, source), golds_set) in
        candidates.iter().zip(sources.iter()).zip(gold_edits.iter())
    {
        let (c, p, g) = score_single(
            candidate,
            source,
            golds_set,
            max_unchanged_words,
            beta,
            ignore_whitespace_casing,
        );
        stat_correct += c;
        stat_proposed += p;
        stat_gold += g;
    }

    let p = if stat_proposed == 0.0 {
        1.0
    } else {
        stat_correct / stat_proposed
    };
    let r = if stat_gold == 0.0 {
        1.0
    } else {
        stat_correct / stat_gold
    };
    let f1 = {
        let sqbeta = beta * beta;
        let denom = sqbeta * p + r;
        if denom == 0.0 {
            0.0
        } else {
            (1.0 + sqbeta) * p * r / denom
        }
    };

    Ok((p, r, f1))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_multi_pre_rec_f1, m)?)?;
    Ok(())
}
