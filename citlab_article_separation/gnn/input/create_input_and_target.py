import json
import numpy as np

from citlab_article_separation.gnn.input.feature_generation import build_input_and_target_bc


def get_input_and_target_from_xml(path_to_xml, provide_relations_to_consider, external_data,
                                  interaction, visual_regions):
    # build input & target
    num_nodes, interacting_nodes, num_interacting_nodes, node_features, edge_features, \
    visual_regions_nodes, num_points_visual_regions_nodes, \
    visual_regions_edges, num_points_visual_regions_edges, \
    gt_relations, gt_num_relations = \
        build_input_and_target_bc(page_path=path_to_xml,
                                  interaction=interaction,
                                  visual_regions=visual_regions,
                                  external_data=external_data)

    # build output dict
    return_dict = dict()
    return_dict["num_nodes"] = num_nodes
    return_dict['interacting_nodes'] = interacting_nodes
    return_dict['num_interacting_nodes'] = num_interacting_nodes
    return_dict['node_features'] = node_features
    return_dict['edge_features'] = edge_features
    if visual_regions_nodes is not None and num_points_visual_regions_nodes is not None:
        return_dict['visual_regions_nodes'] = visual_regions_nodes
        return_dict['num_points_visual_regions_nodes'] = num_points_visual_regions_nodes
    if visual_regions_edges is not None and num_points_visual_regions_edges is not None:
        return_dict['visual_regions_edges'] = visual_regions_edges
        return_dict['num_points_visual_regions_edges'] = num_points_visual_regions_edges

    return_dict['gt_relations'] = gt_relations
    return_dict['gt_num_relations'] = gt_num_relations

    # if provide_relations_to_consider:
    #     return_dict['relations_to_consider'] = np.array(data['relations_to_consider'], dtype=np.int32)
    #     return_dict['num_relations_to_consider'] = np.array(data['num_relations_to_consider'], dtype=np.int32)
    return return_dict


def get_input_and_target_from_json(path_to_json, provide_relations_to_consider):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)

    # get input and targets
    return_dict = dict()
    return_dict['num_nodes'] = np.array(data['num_nodes'], dtype=np.int32)
    return_dict['interacting_nodes'] = np.array(data['interacting_nodes'], dtype=np.int32)
    return_dict['num_interacting_nodes'] = np.array(data['num_interacting_nodes'], dtype=np.int32)
    return_dict['node_features'] = np.array(data['node_features'], dtype=np.float32)
    return_dict['edge_features'] = np.array(data['edge_features'], dtype=np.float32)
    if 'visual_regions_nodes' in data and 'num_points_visual_regions_nodes' in data:
        num_points_visual_regions_nodes = np.array(data['num_points_visual_regions_nodes'], dtype=np.int32)
        # max_num_points_visual_regions_nodes = np.amax(num_points_visual_regions_nodes)
        region_list = []
        for i in range(data['num_nodes']):
            visual_region = np.array(data['visual_regions_nodes'][i], dtype=np.float32)
            # num_points_pad = max_num_points_visual_regions_nodes - num_points_visual_regions_nodes[i]
            # region_list.append(np.array(np.pad(visual_region, ((0, 0), (0, num_points_pad)), mode='constant'), dtype=np.float32))
            region_list.append(visual_region)
        return_dict['num_points_visual_regions_nodes'] = num_points_visual_regions_nodes
        return_dict['visual_regions_nodes'] = np.stack(region_list)
    if 'visual_regions_edges' in data and 'num_points_visual_regions_edges' in data:
        num_points_visual_regions_edges = np.array(data['num_points_visual_regions_edges'], dtype=np.int32)
        # max_num_points_visual_regions_edges = np.amax(num_points_visual_regions_edges)
        region_list = []
        for i in range(data['num_interacting_nodes']):
            visual_region = np.array(data['visual_regions_edges'][i], dtype=np.float32)
            # num_points_pad = max_num_points_visual_regions_edges - num_points_visual_regions_edges[i]
            # region_list.append(np.array(np.pad(visual_region, ((0, 0), (0, num_points_pad)), mode='constant'), dtype=np.float32))
            region_list.append(visual_region)
        return_dict['num_points_visual_regions_edges'] = num_points_visual_regions_edges
        return_dict['visual_regions_edges'] = np.stack(region_list)

    gt_relations = data['gt_relations']
    gt_num_relations = data['gt_num_relations']
    return_dict['gt_relations'] = np.array(gt_relations, dtype=np.int32)
    return_dict['gt_num_relations'] = np.array(gt_num_relations, dtype=np.int32)

    # print("num_nodes ", num_nodes)
    # print("interacting_nodes_shape ", interacting_nodes.shape)
    # print("num_interacting_nodes ", num_interacting_nodes)
    # print("node_features_shape ", node_features.shape)
    # print("gt_relations_shape ", gt_relations.shape)
    # print("gt_num_relations ", gt_num_relations)

    if provide_relations_to_consider:
        return_dict['relations_to_consider'] = np.array(data['relations_to_consider'], dtype=np.int32)
        return_dict['num_relations_to_consider'] = np.array(data['num_relations_to_consider'], dtype=np.int32)

    return return_dict


if __name__ == '__main__':
    lst_path = "/home/johannes/devel/projects/tf_rel/lists/onb+nlf/onb+nlf_textblocks_json16d1.lst"
    with open(lst_path, "r") as lst_file:
        stroke_widths = []
        text_heights = []
        for file in lst_file:
            result = get_input_and_target_from_json(file.strip(), False)
            for i in range(result['node_features'].shape[0]):
                if np.isnan(result['node_features'][i, 14]):
                    print("Set NaN to 0.0")
                    result['node_features'][i, 14] = 0.0
            if np.max(result['node_features'][:, 14]) > 1.5:
                print(file.rstrip())
            stroke_widths.append(result['node_features'][:, 14])
            text_heights.append(result['node_features'][:, 15])
        stroke_widths = np.concatenate(stroke_widths, axis=0)
        text_heights = np.concatenate(text_heights, axis=0)
        print(f"mean stroke_widths = {np.mean(stroke_widths)}")
        print(f"min stroke_widths =  {np.min(stroke_widths[stroke_widths > 0])}")
        print(f"max stroke_widths =  {np.max(stroke_widths)}")
        print(f"mean text_heights =  {np.mean(text_heights)}")
        print(f"min text_heights =   {np.min(text_heights[text_heights > 0])}")
        print(f"max text_heights =   {np.max(text_heights)}")

    # page_path = "/home/johannes/devel/data/NewsEye_GT/AS_BC/NewsEye_NLF_200_textblocks/330069/1872_01_05/page/pr-00001.xml"
    # result = get_input_and_target_from_xml(page_path, False, 16, 1, 'delaunay')
    # print(result['node_features'][:, 14])
