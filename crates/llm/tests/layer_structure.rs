use std::path::PathBuf;

use my_nn_engine::graph::Graph;
use my_nn_engine::graph::ValueId;
use my_nn_engine::tensor::data::TensorData;
use my_nn_engine::tensor::types::SIntType;
use my_nn_engine_llm::build_decoder;
use my_nn_engine_llm::BuildOptions;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::ModelSpec;
use my_nn_engine_llm::ProjStructure;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tiny-llama-random")
}

fn value_name(graph: &Graph, v: ValueId) -> String {
    graph.values[v].name.clone()
}

fn i64_values(graph: &Graph, v: ValueId) -> Vec<i64> {
    match graph.get_initializer(v).unwrap().data {
        TensorData::SInt(SIntType::I64, values) => values,
        other => panic!("expected i64 initializer, got {other:?}"),
    }
}

fn dims(graph: &Graph, v: ValueId) -> Vec<usize> {
    let ty = graph.get_resolved_tensor_type(v).unwrap();
    (0..ty.dims.ndim()).map(|i| ty.dims[i]).collect()
}

fn assert_proj(graph: &Graph, proj: &ProjStructure, weight_name: &str, output_name: &str) {
    assert_eq!(value_name(graph, proj.weight), weight_name);
    assert!(graph.get_external_ref(proj.weight).is_some());
    assert!(proj.scale.is_none(), "f32 model has no weight scale");
    assert!(proj.bias.is_none(), "llama has no bias");
    assert_eq!(value_name(graph, proj.output), output_name);
}

#[test]
fn layer_structure_points_at_projections_and_constants() {
    let dir = model_dir();
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_dir(&dir).unwrap();
    let spec = ModelSpec::from_hf(&config, &hf).unwrap();

    let r = build_decoder(&config, &spec, 64, &BuildOptions::default());
    let graph = &r.graph;

    // tiny-llama-random: 2 layers, 4 heads, 4 kv heads, head_dim 4; decode mode
    // means seq_q = 1.
    assert_eq!(r.layers.len(), 2);
    for (i, layer) in r.layers.iter().enumerate() {
        let p = format!("model.layers.{i}");
        let attn = &layer.attention;
        assert_proj(
            graph,
            &attn.q,
            &format!("{p}.self_attn.q_proj.weight"),
            &format!("{p}_q_proj"),
        );
        assert_proj(
            graph,
            &attn.k,
            &format!("{p}.self_attn.k_proj.weight"),
            &format!("{p}_k_proj"),
        );
        assert_proj(
            graph,
            &attn.v,
            &format!("{p}.self_attn.v_proj.weight"),
            &format!("{p}_v_proj"),
        );
        assert_proj(
            graph,
            &attn.o,
            &format!("{p}.self_attn.o_proj.weight"),
            &format!("{p}_o_proj"),
        );
        assert_proj(
            graph,
            &layer.mlp.gate,
            &format!("{p}.mlp.gate_proj.weight"),
            &format!("{p}_gate_proj"),
        );
        assert_proj(
            graph,
            &layer.mlp.up,
            &format!("{p}.mlp.up_proj.weight"),
            &format!("{p}_up_proj"),
        );
        assert_proj(
            graph,
            &layer.mlp.down,
            &format!("{p}.mlp.down_proj.weight"),
            &format!("{p}_down_proj"),
        );

        assert_eq!(i64_values(graph, attn.q_shape), vec![1, 1, 4, 4]);
        assert_eq!(i64_values(graph, attn.kv_shape), vec![1, 1, 4, 4]);
        assert_eq!(i64_values(graph, attn.attn_out_shape), vec![1, 1, 16]);

        assert_eq!(value_name(graph, attn.k_cache), format!("{p}.past_key"));
        assert_eq!(value_name(graph, attn.v_cache), format!("{p}.past_value"));
        assert_eq!(dims(graph, attn.k_cache), vec![1, 4, 64, 4]);
        assert_eq!(dims(graph, attn.v_cache), vec![1, 4, 64, 4]);
        assert!(attn.k_cache_scale.is_none() && attn.v_cache_scale.is_none());
    }
}
