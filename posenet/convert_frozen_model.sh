tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='output_raw_heatmaps,output_raw_offsets,output_raw_part_heatmaps,output_raw_segments' \
    --saved_model_tags=serve \
    ./mobilenet_v1_100_kpt_seg_parts_stripped.pb \
./
