# -*- coding: utf-8 -*-

import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

def select_diverse_nodes(graph, num_nodes):
    """
    智能选择节点,确保涵盖所有节点类型。
    
    参数:
        graph: NetworkX图对象
        num_nodes: 需要选择的节点数量
        
    返回:
        list: 选中的节点列表
    """
    if len(graph.nodes) == 0:
        return []
    
    # 获取所有节点的node_type属性
    node_types = nx.get_node_attributes(graph, "node_type")
    
    # 如果没有node_type属性,直接返回前num_nodes个节点
    if not node_types:
        print("警告: 图中节点没有 'node_type' 属性,使用顺序选择策略。")
        return list(graph.nodes)[:num_nodes]
    
    # 按节点类型分组
    type_to_nodes = defaultdict(list)
    for node, node_type in node_types.items():
        type_to_nodes[node_type].append(node)
    
    unique_types = list(type_to_nodes.keys())
    num_unique_types = len(unique_types)
    
    print(f"检测到 {num_unique_types} 种节点类型: {unique_types}")
    
    # 如果请求的节点数大于等于总节点数,返回所有节点
    if num_nodes >= len(graph.nodes):
        return list(graph.nodes)
    
    # 策略1: 如果节点类型数量 <= 请求节点数,确保每种类型至少有一个代表
    if num_unique_types <= num_nodes:
        selected_nodes = []
        
        # 第一步: 从每种类型中至少选择一个节点
        for node_type in unique_types:
            nodes_of_type = type_to_nodes[node_type]
            # 选择该类型的第一个节点(可以改为随机选择)
            selected_nodes.append(nodes_of_type[0])
        
        remaining_slots = num_nodes - len(selected_nodes)
        
        # 第二步: 如果有剩余名额,按比例分配给各类型
        if remaining_slots > 0:
            # 计算每种类型的节点数量占比
            total_nodes = sum(len(nodes) for nodes in type_to_nodes.values())
            
            # 按比例分配剩余名额
            additional_selections = {}
            for node_type in unique_types:
                proportion = len(type_to_nodes[node_type]) / total_nodes
                additional_selections[node_type] = max(1, int(remaining_slots * proportion))
            
            # 调整总数使其等于remaining_slots
            total_additional = sum(additional_selections.values())
            if total_additional != remaining_slots:
                # 微调: 对最大的类型进行调整
                diff = remaining_slots - total_additional
                largest_type = max(unique_types, key=lambda t: len(type_to_nodes[t]))
                additional_selections[largest_type] += diff
            
            # 第三步: 为每种类型选择额外的节点
            for node_type, count in additional_selections.items():
                nodes_of_type = type_to_nodes[node_type]
                # 排除已选择的节点
                already_selected = [n for n in selected_nodes if n in nodes_of_type]
                available_nodes = [n for n in nodes_of_type if n not in already_selected]
                
                # 选择额外的节点(取前count个,避免重复)
                extra_nodes = available_nodes[:min(count, len(available_nodes))]
                selected_nodes.extend(extra_nodes)
        
        # 确保不超过请求的节点数
        selected_nodes = selected_nodes[:num_nodes]
        
        # 统计最终选择的节点类型分布
        final_type_counts = defaultdict(int)
        for node in selected_nodes:
            node_type = node_types.get(node, "unknown")
            final_type_counts[node_type] += 1
        
        print(f"节点选择结果 (共 {len(selected_nodes)} 个节点):")
        for node_type, count in sorted(final_type_counts.items()):
            original_count = len(type_to_nodes[node_type])
            print(f"  - {node_type}: {count}/{original_count} 个节点")
        
        return selected_nodes
    
    else:
        # 策略2: 如果节点类型数量 > 请求节点数,优先选择不同类型的节点
        print(f"警告: 节点类型数量 ({num_unique_types}) 超过请求节点数 ({num_nodes}),无法涵盖所有类型。")
        print("将尝试选择最具代表性的节点类型...")
        
        # 按节点数量排序,优先选择节点多的类型
        sorted_types = sorted(unique_types, key=lambda t: len(type_to_nodes[t]), reverse=True)
        
        selected_nodes = []
        for node_type in sorted_types[:num_nodes]:
            nodes_of_type = type_to_nodes[node_type]
            selected_nodes.append(nodes_of_type[0])
        
        return selected_nodes[:num_nodes]


def visualize_graph(graph_path, num_nodes=30):
    """
    读取保存的图并可视化部分节点。

    参数：
        graph_path (str): 图文件的路径。
        num_nodes (int): 要可视化的节点数量（建议 20-50，避免重叠）。
    """
    if not os.path.exists(graph_path):
        print(f"图文件 {graph_path} 不存在！")
        return

    try:
        # 加载图，增加 weights_only=False 以兼容加载自定义对象（如 nx.Graph）
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)

        # 检查加载的对象是否是 networkx 图
        if not isinstance(graph, nx.Graph):
            print(f"警告: 加载的对象类型是 {type(graph)}，尝试继续...")
        
        # 提取子图
        if len(graph.nodes) == 0:
            print("图中没有节点，无法可视化。")
            return

        # 使用智能节点选择策略,确保涵盖所有节点类型
        sub_nodes = select_diverse_nodes(graph, num_nodes)
        actual_num_nodes = len(sub_nodes)
        
        if actual_num_nodes == 0:
            print("未能选择任何节点,无法可视化。")
            return
            
        subgraph = graph.subgraph(sub_nodes)
        
        print(f"原始图节点数: {len(graph.nodes)}, 可视化节点数: {actual_num_nodes}")

        # 根据节点数量动态调整画布大小
        fig_width = max(12, actual_num_nodes * 0.4)
        fig_height = max(8, actual_num_nodes * 0.3)
        plt.figure(figsize=(fig_width, fig_height))
        
        # 检测连通分量（兼容有向图和无向图）
        if isinstance(subgraph, (nx.DiGraph, nx.MultiDiGraph)):
            # 有向图使用弱连通分量
            connected_components = list(nx.weakly_connected_components(subgraph))
        else:
            # 无向图使用标准连通分量
            connected_components = list(nx.connected_components(subgraph))
        
        num_components = len(connected_components)
        
        if num_components > 1:
            print(f"检测到 {num_components} 个连通分量，使用优化的布局策略...")
            
            # 找到最大的连通分量
            largest_component = max(connected_components, key=len)
            largest_subgraph = subgraph.subgraph(largest_component)
            
            # 为最大连通分量使用 spring_layout
            pos_largest = nx.spring_layout(
                largest_subgraph, 
                seed=42, 
                k=2.0 / (len(largest_component) ** 0.5),
                iterations=100
            )
            
            # 为其他小分量分配位置
            pos_other = {}
            offset_x = max(pos_largest.values(), key=lambda p: p[0])[0] + 3.0 if pos_largest else 0
            offset_y = 0
            
            for i, component in enumerate(connected_components):
                if component == largest_component:
                    continue
                
                comp_subgraph = subgraph.subgraph(component)
                if len(component) == 1:
                    # 孤立节点
                    node = list(component)[0]
                    pos_other[node] = (offset_x, offset_y)
                    offset_y += 2.0
                else:
                    # 小连通分量
                    comp_pos = nx.spring_layout(
                        comp_subgraph, 
                        seed=42, 
                        k=1.5 / (len(component) ** 0.5),
                        iterations=50
                    )
                    # 平移位置
                    for node, (x, y) in comp_pos.items():
                        pos_other[node] = (x + offset_x, y + offset_y)
                    offset_x += 3.0
                    offset_y = 0
            
            # 合并位置
            pos = {**pos_largest, **pos_other}
        else:
            # 单个连通分量，使用标准布局
            pos = nx.spring_layout(
                subgraph, 
                seed=42, 
                k=2.0 / (actual_num_nodes ** 0.5),  # 动态调整斥力系数
                iterations=100  # 增加迭代次数
            )
        
        # 根据节点数量动态调整节点和字体大小
        node_size = max(200, 800 - actual_num_nodes * 10)
        font_size = max(6, 10 - actual_num_nodes // 10)
        
        # 检查是否有节点类型标签
        node_labels = nx.get_node_attributes(subgraph, "node_type")
        has_node_type_labels = bool(node_labels)
        
        # 如果有节点类型标签，使用它们；否则使用节点 ID
        if has_node_type_labels:
            # 只绘制节点类型标签，不显示节点 ID
            nx.draw(
                subgraph,
                pos,
                with_labels=False,  # 不显示节点 ID
                node_size=node_size,
                node_color="skyblue",
                font_color="black",
                edge_color="gray",
                width=1.5,
                alpha=0.8
            )
            # 单独绘制节点类型标签
            nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=font_size)
        else:
            # 没有节点类型标签时，显示节点 ID
            nx.draw(
                subgraph,
                pos,
                with_labels=True,  # 显示节点 ID
                node_size=node_size,
                node_color="skyblue",
                font_size=font_size,
                font_color="black",
                edge_color="gray",
                width=1.5,
                alpha=0.8
            )

        # 添加边标签（多重图只绘制第一条边的标签）
        if isinstance(subgraph, (nx.MultiGraph, nx.MultiDiGraph)):
            # 对于多重图，只保留每对节点间第一条边的标签
            try:
                edge_label_dict = {}
                for u, v, key, data in subgraph.edges(keys=True, data=True):
                    if 'label' in data:
                        edge_key = (u, v)
                        # 只在第一次遇到该节点对时记录标签（即第一条边）
                        if edge_key not in edge_label_dict:
                            edge_label_dict[edge_key] = data['label']
                
                if edge_label_dict:
                    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_label_dict, font_size=6)
            except Exception as label_error:
                print(f"警告: 绘制边标签时出错 - {label_error}，已跳过。")
        else:
            # 对于简单图，直接绘制
            edge_labels = nx.get_edge_attributes(subgraph, "label")
            if edge_labels:
                try:
                    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)
                except Exception as label_error:
                    print(f"警告: 绘制边标签时出错 - {label_error}")

        plt.title(f"Graph Visualization: {os.path.basename(graph_path)}")
        plt.axis("off")
        
        # 保存图像到统一的输出目录
        output_base_dir = "/home/pids/artifacts/construction/optc_h501/visualizations"
        
        # 确保输出目录存在
        try:
            os.makedirs(output_base_dir, exist_ok=True)
        except Exception as mkdir_error:
            print(f"警告: 创建输出目录失败 - {mkdir_error}")
            output_base_dir = os.path.dirname(graph_path)  # 回退到原目录
        
        base_name = os.path.basename(graph_path)
        # 将文件名中的特殊字符替换为下划线，避免路径问题
        safe_name = base_name.replace(" ", "_").replace("~", "_").replace(":", "_")
        output_path = os.path.join(output_base_dir, f"visualization_{safe_name}.png")
        
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"✅ 图像已保存到: {output_path}")
        except Exception as save_error:
            print(f"警告: 保存图像时出错 - {save_error}")
        
        # 尝试显示图像（如果有图形界面）
        try:
            plt.show()
        except Exception:
            print("提示: 无法显示图形窗口，请查看保存的 PNG 文件。")
        finally:
            plt.close()  # 关闭图形以释放内存
        
    except Exception as e:
        print(f"加载或可视化文件 {graph_path} 时出错: {e}")

def list_graph_files(directory):
    """
    列出指定目录中的所有图文件。

    参数：
        directory (str): 图文件所在的目录。

    返回：
        list: 图文件路径列表。
    """
    # 获取目录下所有文件，因为实际文件可能没有 .pt 后缀
    if not os.path.isdir(directory):
        print(f"目录 {directory} 不存在或不是一个有效的目录！")
        return []
    
    all_files = os.listdir(directory)
    # 过滤出完整路径，并可选地过滤掉隐藏文件或目录
    graph_files = [
        os.path.join(directory, f) 
        for f in all_files 
        if os.path.isfile(os.path.join(directory, f))
    ]
    
    if not graph_files:
        print(f"目录 {directory} 中没有找到任何文件！")
    else:
        print(f"在目录 {directory} 中找到 {len(graph_files)} 个文件。")
        
    return graph_files

def main():
    """
    主函数：列出图文件并选择可视化。
    """
    # 确定的 Linux 系统图文件目录路径
    graph_dir = "/home/pids/artifacts/construction/optc_h501/construction/c3f3165f1d82cab33612a1436909d57d0846d69379ace2964781ab158f0656c2/nx/graph_2019-09-24"

    # 列出所有图文件
    graph_files = list_graph_files(graph_dir)
    if not graph_files:
        return

    print("找到以下图文件：")
    for idx, file in enumerate(graph_files):
        print(f"[{idx}] {os.path.basename(file)}")

    # 选择要可视化的文件
    try:
        selected_indices_input = input("请输入要可视化的文件索引（用逗号分隔，例如 0,1,2）：")
        if not selected_indices_input.strip():
            print("未输入索引，退出。")
            return
        selected_indices = [int(idx.strip()) for idx in selected_indices_input.split(",")]
    except ValueError:
        print("输入无效，请输入有效的整数索引！")
        return
    
    # 询问用户希望的节点数量
    try:
        num_nodes_input = input("请输入要可视化的节点数量（默认 30，建议 20-50）：").strip()
        if num_nodes_input:
            custom_num_nodes = int(num_nodes_input)
            if custom_num_nodes < 1:
                print("节点数量必须大于 0，使用默认值 30。")
                custom_num_nodes = 30
        else:
            custom_num_nodes = 30
    except ValueError:
        print("输入无效，使用默认值 30。")
        custom_num_nodes = 30

    for idx in selected_indices:
        if 0 <= idx < len(graph_files):
            print(f"正在可视化文件：{graph_files[idx]}")
            visualize_graph(graph_files[idx], num_nodes=custom_num_nodes)
        else:
            print(f"索引 {idx} 超出范围（0-{len(graph_files)-1}）！")

if __name__ == "__main__":
    main()