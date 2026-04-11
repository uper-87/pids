# -*- coding: utf-8 -*-

import torch
import networkx as nx
import matplotlib.pyplot as plt
import os
import glob

def visualize_graph(graph_path, num_nodes=50):
    """
    读取保存的图并可视化部分节点。

    参数：
        graph_path (str): 图文件的路径。
        num_nodes (int): 要可视化的节点数量。
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

        sub_nodes = list(graph.nodes)[:num_nodes]  # 选择前 num_nodes 个节点
        subgraph = graph.subgraph(sub_nodes)

        # 绘制图
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(subgraph, seed=42)  # 使用 spring 布局，固定种子以便复现
        nx.draw(
            subgraph,
            pos,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            font_size=8,
            font_color="black",
            edge_color="gray",
        )

        # 添加节点类型标签
        node_labels = nx.get_node_attributes(subgraph, "node_type")
        if node_labels:
            nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8)

        # 添加边标签
        edge_labels = nx.get_edge_attributes(subgraph, "label")
        if edge_labels:
            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)

        plt.title(f"Graph Visualization: {os.path.basename(graph_path)}")
        plt.axis("off")
        plt.show()
        
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

    for idx in selected_indices:
        if 0 <= idx < len(graph_files):
            print(f"正在可视化文件：{graph_files[idx]}")
            visualize_graph(graph_files[idx], num_nodes=50)
        else:
            print(f"索引 {idx} 超出范围（0-{len(graph_files)-1}）！")

if __name__ == "__main__":
    main()