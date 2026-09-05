//! Fast Rust core for M2 scorer (Grammatical Error Correction evaluation).
//!
//! Replicates Dahlmeier & Ng (2012) MaxMatch evaluation with exact parity,
//! while eliminating quadratic bottlenecks and redundant path relaxations.

use pyo3::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum EditKind {
    Ins,
    Del,
    Sub,
    Noop,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Edit {
    kind: EditKind,
    start: usize,
    end: usize,
    orig: String,
    corr: String,
    unchanged_words: usize,
}

#[derive(Clone, Debug)]
struct GoldEdit {
    start: i64,
    end: i64,
    orig: String,
    corrections: Vec<String>,
}

type Vertex = (usize, usize);
type Edge = (Vertex, Vertex);

struct EditGraph {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    dist: HashMap<Edge, f64>,
    edits: HashMap<Edge, Edit>,
}

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
            end: 0,
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
                    end: i,
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
    let mut visited: HashSet<Vertex> = HashSet::new();

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

fn merge_graphs(g1: EditGraph, g2: EditGraph) -> EditGraph {
    let mut vertex_set: HashSet<Vertex> = g1.vertices.iter().copied().collect();
    let mut vertices = g1.vertices;
    for v in g2.vertices {
        if vertex_set.insert(v) {
            vertices.push(v);
        }
    }
    vertices.sort_unstable();

    let mut edge_set: HashSet<Edge> = g1.edges.iter().copied().collect();
    let mut edges = g1.edges;
    for e in g2.edges {
        if edge_set.insert(e) {
            edges.push(e);
        }
    }
    edges.sort_unstable();

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
        (EditKind::Ins, EditKind::Sub | EditKind::Noop) => (
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
        (EditKind::Del, EditKind::Sub | EditKind::Noop) => (
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
        (EditKind::Sub, EditKind::Sub | EditKind::Noop) => (
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

fn transitive_arcs(mut graph: EditGraph, max_unchanged_words: usize) -> EditGraph {
    let v = graph.vertices.clone();
    let n = v.len();

    let mut existing_edges: HashSet<Edge> = graph.edges.iter().copied().collect();

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
                        if existing_edges.insert((vi, vj)) {
                            graph.edges.push((vi, vj));
                        }
                        graph.dist.insert((vi, vj), dik + dkj);
                        graph.edits.insert((vi, vj), merged);
                    }
                }
            }
        }
    }

    // Identify transitive noop arcs to prune
    let to_remove: HashSet<Edge> = graph
        .edges
        .iter()
        .filter(|&&e| {
            graph
                .edits
                .get(&e)
                .map_or(false, |ed| ed.kind == EditKind::Noop)
                && *graph.dist.get(&e).unwrap_or(&0.0) > 1.0
        })
        .copied()
        .collect();

    // Fast O(|E|) batch deletion
    graph.edges.retain(|e| !to_remove.contains(e));
    for e in to_remove {
        graph.dist.insert(e, f64::INFINITY);
        graph.edits.remove(&e);
    }

    graph
}

fn edit_matches_gold(edit: &Edit, gold: &GoldEdit) -> bool {
    edit.start == gold.start as usize
        && edit.end == gold.end as usize
        && edit.orig == gold.orig
        && gold.corrections.contains(&edit.corr)
}

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
        edges.sort_unstable();
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
            let mut g_rptr = if gold_list.is_empty() {
                0
            } else {
                gold_list.len() - 1
            };

            while lptr <= rptr {
                let edge = span_edges[cur];
                let this_edit = match graph.edits.get(&edge) {
                    Some(e) => e,
                    None => break,
                };

                let cur_gold_range: Vec<usize> = if gold_list.is_empty() {
                    Vec::new()
                } else if cur == lptr {
                    (g_lptr..=g_rptr.min(gold_list.len() - 1)).collect()
                } else {
                    (g_lptr..=g_rptr.min(gold_list.len() - 1)).rev().collect()
                };

                let mut has_match = false;
                for &i in &cur_gold_range {
                    if i < gold_list.len() && edit_matches_gold(this_edit, gold_list[i]) {
                        has_match = true;
                        ret_dist.insert(edge, -num_edges);
                        if cur == lptr {
                            g_lptr = i + 1;
                        } else if i > 0 {
                            g_rptr = i - 1;
                        } else {
                            g_rptr = 0;
                        }
                        break;
                    }
                }

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

fn best_edit_seq_bf(graph: &EditGraph, dist: &HashMap<Edge, f64>) -> Vec<Edit> {
    let mut this_dist: HashMap<Vertex, f64> = HashMap::new();
    let mut path: HashMap<Vertex, Vertex> = HashMap::new();

    for &v in &graph.vertices {
        this_dist.insert(v, f64::INFINITY);
    }
    this_dist.insert((0, 0), 0.0);

    // Bellman-Ford with early stopping
    for _ in 0..graph.vertices.len().saturating_sub(1) {
        let mut updated = false;
        for &(vi, vw) in &graph.edges {
            let d_vi = *this_dist.get(&vi).unwrap_or(&f64::INFINITY);
            if d_vi.is_infinite() {
                continue;
            }
            let d_edge = *dist.get(&(vi, vw)).unwrap_or(&f64::INFINITY);
            let d_vw = *this_dist.get(&vw).unwrap_or(&f64::INFINITY);
            if d_vi + d_edge < d_vw {
                this_dist.insert(vw, d_vi + d_edge);
                path.insert(vw, vi);
                updated = true;
            }
        }
        if !updated {
            break;
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

fn equals_ignore_whitespace_casing(a: &str, b: &str) -> bool {
    a.replace(' ', "").to_lowercase() == b.replace(' ', "").to_lowercase()
}

#[inline]
fn comp_p(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        1.0
    } else {
        a / b
    }
}

#[inline]
fn comp_r(c: f64, g: f64) -> f64 {
    if g == 0.0 {
        1.0
    } else {
        c / g
    }
}

#[inline]
fn comp_f1(c: f64, e: f64, g: f64, beta: f64) -> f64 {
    let sqbeta = beta * beta;
    let denom = sqbeta * g + e;
    if denom == 0.0 {
        if c == 0.0 {
            1.0
        } else {
            0.0
        }
    } else {
        (1.0 + sqbeta) * c / denom
    }
}

/// Scores a batch of hypothesis sentences against gold M2 annotations.
///
/// Annotator selection is evaluated cumulatively per sentence matching the
/// official Dahlmeier & Ng Python implementation.
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

    let gold_edits: Vec<BTreeMap<u32, Vec<GoldEdit>>> = gold_edits_raw
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
    let sqbeta = beta * beta;

    for ((candidate, source), golds_set) in
        candidates.iter().zip(sources.iter()).zip(gold_edits.iter())
    {
        let candidate_tok: Vec<String> = candidate.split_whitespace().map(String::from).collect();
        let source_tok: Vec<String> = source.split_whitespace().map(String::from).collect();

        let (mat1, bp1) = levenshtein_matrix(&source_tok, &candidate_tok, 1, 1, 1);
        let (mat2, bp2) = levenshtein_matrix(&source_tok, &candidate_tok, 1, 1, 2);

        let g1 = build_edit_graph(&mat1, &bp1);
        let g2 = build_edit_graph(&mat2, &bp2);
        let mut graph = merge_graphs(g1, g2);
        graph = transitive_arcs(graph, max_unchanged_words);

        let mut f1_max = -1.0f64;
        let mut max_stat_correct = -1.0f64;
        let mut min_stat_proposed = f64::INFINITY;
        let mut min_stat_gold = f64::INFINITY;

        let mut argmax_correct = 0.0f64;
        let mut argmax_proposed = 0.0f64;
        let mut argmax_gold = 0.0f64;

        // BTreeMap guarantees deterministic annotator iteration
        for gold_list in golds_set.values() {
            let local_dist = set_weights(&graph, gold_list);
            let mut edit_seq = best_edit_seq_bf(&graph, &local_dist);

            if ignore_whitespace_casing {
                edit_seq.retain(|e| !equals_ignore_whitespace_casing(&e.orig, &e.corr));
            }

            let correct = match_seq(&edit_seq, gold_list);

            let stat_correct_local = stat_correct + correct.len() as f64;
            let stat_proposed_local = stat_proposed + edit_seq.len() as f64;
            let stat_gold_local = stat_gold + gold_list.len() as f64;

            let f1_local = comp_f1(
                stat_correct_local,
                stat_proposed_local,
                stat_gold_local,
                beta,
            );

            let is_better = f1_max < f1_local
                || (f1_max == f1_local && max_stat_correct < stat_correct_local)
                || (f1_max == f1_local
                    && max_stat_correct == stat_correct_local
                    && min_stat_proposed + sqbeta * min_stat_gold
                        > stat_proposed_local + sqbeta * stat_gold_local);

            if is_better {
                f1_max = f1_local;
                max_stat_correct = stat_correct_local;
                min_stat_proposed = stat_proposed_local;
                min_stat_gold = stat_gold_local;
                argmax_correct = correct.len() as f64;
                argmax_proposed = edit_seq.len() as f64;
                argmax_gold = gold_list.len() as f64;
            }
        }

        stat_correct += argmax_correct;
        stat_proposed += argmax_proposed;
        stat_gold += argmax_gold;
    }

    let p = comp_p(stat_correct, stat_proposed);
    let r = comp_r(stat_correct, stat_gold);
    let f1 = {
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
