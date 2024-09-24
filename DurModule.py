import torch
import numpy as np


# 寻找最相似的类别（除了当前类别之外，平均logit分数最大的别的非当前头（尾部）类别）
def search_most_similar_class(args, pred_all_tensor, label_all_tensor, logits_all_tensor):
    tail_classes = args.tail_classes
    # 计算每一个类别的平均预测分数
    average_scores = torch.zeros(args.num_classes, args.num_classes)

    # 遍历每一个类别 计算该类别的平均预测分数
    for c in range(args.num_classes):
        class_mask = (label_all_tensor == c)
        class_logits = pred_all_tensor[class_mask]  # 当前类别所有的logits
        # 计算所有logits的平均值
        average_avg_scores = class_logits.mean(dim=0)
        average_scores[c] = average_avg_scores

    # 寻找最相似的类别（除了当前类别之外，平均logit分数最大的别的非当前头（尾部）类别）
    most_sim_classes = []
    for c in range(args.num_classes):
        avg_scores_c = average_scores[c]
        avg_scores_c[c] = float('-inf')  # 当前类别置为负无穷大，不参与比较
        if c in tail_classes:
            # c是尾部类，将所有尾部类设置为负无穷大 不参与比较
            for c_idx in range(args.num_classes):
                if c_idx in tail_classes:
                    avg_scores_c[c_idx] = float('-inf')
        else:
            # c是头部类，将所有的头部类都设置为负无穷大 不参与比较
            for c_idx in range(args.num_classes):
                if c_idx not in tail_classes:
                    avg_scores_c[c_idx] = float('-inf')
        most_similar_class = torch.argmax(avg_scores_c).item()  # 最相似类别
        most_sim_classes.append(most_similar_class)
        print(f"Class {c} is most similar to class {most_similar_class}")

    print("Most similar classes for each class:", most_sim_classes)

    # 相似类的logits集合 后续利用它们指导尾部类的分布恢复
    # [[logits_1, logits_2], ...]
    # 如果是头部类 logits_1 是头部类 logits_2 是尾部类；
    # 如果是尾部类 logits_1 是尾部类 logits_2 是头部类
    sim_logits_matrix = []  # [[shape(m, 128), shape(n, 128)],...]
    for c in range(args.num_classes):
        class_mask = (label_all_tensor == c)
        class_sim_mask = (label_all_tensor == most_sim_classes[c])
        src_class_logits = logits_all_tensor[class_mask]
        tar_class_logits = logits_all_tensor[class_sim_mask]
        # print(f"class {c} is src_class_logits : {src_class_logits.shape}, tar_class_logits : {tar_class_logits.shape}")
        sim_logits_matrix.append([src_class_logits, tar_class_logits])

    return most_sim_classes

# 计算与目标类别最相似的类别的所有特征值、所有特征向量、几何相似度
# 这里后续计算不需要用到几何相似度，原因作者已经验证：我们已经使用了类别相似度来获得最相似的类别
# 如果两个类别之间的相似度较高，那么它们的特征分布几何结构也表现出较高的相似性。而随着类别相似度的降低，类别特征分布几何结构之间的相似性呈下降趋势
def get_eigenvalues_and_eigenvectors_and_sim_geometric(args, most_sim_classes, label_all_tensor, logits_all_tensor):
    # 相似类的logits集合 后续利用它们指导尾部类的分布恢复 如果是头部类 则0 是头部类 1 是尾部类；
    # 如果是尾部类 则0 是尾部类 1是头部类
    sim_logits_matrix = []  # [[shape(m, 128), shape(n, 128)],...]
    for c in range(args.num_classes):
        class_mask = (label_all_tensor == c)
        class_sim_mask = (label_all_tensor == most_sim_classes[c])
        src_class_logits = logits_all_tensor[class_mask]
        tar_class_logits = logits_all_tensor[class_sim_mask]
        # print(f"class {c} is src_class_logits : {src_class_logits.shape}, tar_class_logits : {tar_class_logits.shape}")
        sim_logits_matrix.append([src_class_logits, tar_class_logits])

    # 计算他们的协方差矩阵
    sim_geometric = []  # 存储相似度
    sorted_eigenvalues_list = []  # 类别C的特征值[[values], [values],...]
    sorted_eigenvectors_list = []  # 类别c的特征向量 [[128, 128], [128, 128], ...]
    for c in range(args.num_classes):
        src_class_logits = sim_logits_matrix[c][0]
        tar_class_logits = sim_logits_matrix[c][1]
        # 计算协方差矩阵
        src_covariance_matrix = np.cov(src_class_logits.cpu(), rowvar=False)
        tar_covariance_matrix = np.cov(tar_class_logits.cpu(), rowvar=False)

        # print(f"class {c} is src_covariance_matrix : {src_covariance_matrix.shape}, "
        #       f"tar_covariance_matrix : {tar_covariance_matrix.shape}")
        # 进行特征值分解
        src_eigenvalues, src_eigenvectors = np.linalg.eigh(src_covariance_matrix)
        tar_eigenvalues, tar_eigenvectors = np.linalg.eigh(tar_covariance_matrix)

        # 对特征值进行排序
        src_sorted_indices = np.argsort(src_eigenvalues)[::-1]
        src_eigenvalues = src_eigenvalues[src_sorted_indices]
        src_eigenvectors = src_eigenvectors[:, src_sorted_indices]

        src_sorted_indices = np.argsort(tar_eigenvalues)[::-1]
        tar_eigenvalues = tar_eigenvalues[src_sorted_indices]
        tar_eigenvectors = tar_eigenvectors[:, src_sorted_indices]
        # 存储排序以后的特征值和特征向量 主要面向 尾部类 -》头部类，
        # 这里只存储目标类别对应的最相似类别的特征值和特征向量即可
        sorted_eigenvalues_list.append(tar_eigenvalues)
        sorted_eigenvectors_list.append(tar_eigenvectors)

        similarity = 0
        for i in range(len(tar_eigenvectors)):
            similarity += np.abs(np.dot(src_eigenvectors[:, i].T, tar_eigenvectors[:, i]))
        sim_geometric.append(similarity)
        # print("Similarity of the geometric shapes of the two perceptual manifolds:", similarity)

    print(f"similarity of geometric list {sim_geometric}")
    return sorted_eigenvalues_list, sorted_eigenvectors_list, sim_geometric