#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    if (items.empty()) return nullptr;

    // Sort items by the current axis
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    // Choose median as pivot element
    size_t n = items.size();
    size_t median_idx = (n - 1) / 2;
    auto median = items[median_idx];

    // Create node and construct subtrees
    Node* node = new Node{median.first, median.second, nullptr, nullptr};

    // Split items into left and right subarrays
    std::vector<std::pair<Embedding_T, int>> left_items(items.begin(), items.begin() + median_idx);
    std::vector<std::pair<Embedding_T, int>> right_items(items.begin() + median_idx + 1, items.end());
    node->left = buildKD(left_items, depth + 1);
    node->right = buildKD(right_items, depth + 1);
    return node;
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    if (!node) return;

    // Compute distance to the query
    float dist = distance(Node::queryEmbedding, node->embedding);

    // Maintain max-heap of size at most K
    if ((int)heap.size() < K) {
        heap.push({dist, node->idx});
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.push({dist, node->idx});
    }

    // Decide which subtree to search first (near/far)
    bool goLeft = Node::queryEmbedding < node->embedding;
    Node* near = goLeft ? node->left : node->right;
    Node* far = goLeft ? node->right : node->left;

    knnSearch(near, depth + 1, K, heap);

    // Check if we need to search the far subtree
    float splitDist = std::abs(Node::queryEmbedding - node->embedding);
    if ((int)heap.size() < K || splitDist < heap.top().first) {
        knnSearch(far, depth + 1, K, heap);
    }
    return;
}