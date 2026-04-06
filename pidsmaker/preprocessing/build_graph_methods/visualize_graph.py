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

    # 加载图
    graph = torch.load(graph_path)

    # 提取子图
    sub_nodes = list(graph.nodes)[:num_nodes]  # 选择前 num_nodes 个节点
    subgraph = graph.subgraph(sub_nodes)

    # 绘制图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph)  # 使用 spring 布局
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=500,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        edge_color="gray",
    )

    # 添加节点类型标签
    node_labels = nx.get_node_attributes(subgraph, "node_type")
    nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=8)

    # 添加边标签
    edge_labels = nx.get_edge_attributes(subgraph, "label")
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Graph Visualization")
    plt.show()

def list_graph_files(directory):
    """
    列出指定目录中的所有图文件。

    参数：
        directory (str): 图文件所在的目录。

    返回：
        list: 图文件路径列表。
    """
    graph_files = glob.glob(f"{directory}/*.pt")
    if not graph_files:
        print(f"目录 {directory} 中没有找到任何图文件！")
    return graph_files

def main():
    """
    主函数：列出图文件并选择可视化。
    """
    # 修改为实际的图文件目录
    graph_dir = "example_graph_directory"

    # 列出所有图文件
    graph_files = list_graph_files(graph_dir)
    if not graph_files:
        return

    print("找到以下图文件：")
    for idx, file in enumerate(graph_files):
        print(f"[{idx}] {file}")

    # 选择要可视化的文件
    try:
        selected_indices = input("请输入要可视化的文件索引（用逗号分隔，例如 0,1,2）：")
        selected_indices = [int(idx.strip()) for idx in selected_indices.split(",")]
    except ValueError:
        print("输入无效，请输入有效的索引！")
        return

    for idx in selected_indices:
        if 0 <= idx < len(graph_files):
            print(f"正在可视化文件：{graph_files[idx]}")
            visualize_graph(graph_files[idx], num_nodes=50)
        else:
            print(f"索引 {idx} 超出范围！")

if __name__ == "__main__":
    main()