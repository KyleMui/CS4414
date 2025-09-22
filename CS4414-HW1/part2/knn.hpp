#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>
#include <algorithm>  
#include <cmath>      
#include <type_traits> 


template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
};


// dynamic vector: runtime-D (global, set once at startup)
inline size_t& runtime_dim() {
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }
    
    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};


// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}


// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;


/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0)
{
    if (items.empty()) return nullptr;

    // Determine splitting axis for multi-D 
    size_t axis = depth % Embedding_T<T>::Dim();

    // Sort items by the current axis
    std::sort(items.begin(), items.end(), [axis](const auto& a, const auto& b) {
        if (getCoordinate(a.first, axis) != getCoordinate(b.first, axis)) {
            return getCoordinate(a.first, axis) < getCoordinate(b.first, axis);
        }
        return a.second < b.second;
    });

    // Choose median as pivot element 
    size_t n = items.size();
    size_t median_idx = (n - 1) / 2;
    auto median = items[median_idx];

    // Create node and construct subtrees 
    Node<T>* node = new Node<T>{median.first, median.second, nullptr, nullptr};

    // Split items into left and right subarrays 
    std::vector<std::pair<T, int>> left_items(items.begin(), items.begin() + median_idx);
    std::vector<std::pair<T, int>> right_items(items.begin() + median_idx + 1, items.end());
    node->left = buildKD(left_items, depth + 1);
    node->right = buildKD(right_items, depth + 1);
    return node;
}

template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;


/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    if (!node) return;

    float dist = Embedding_T<T>::distance(Node<T>::queryEmbedding, node->embedding);

    // Maintain max-heap of size at most K 
    if ((int)heap.size() < K) {
        heap.push({dist, node->idx});
    } else if (dist < heap.top().first) {
        heap.pop();
        heap.push({dist, node->idx});
    }

    // Determine splitting axis for multi-D
    size_t axis = depth % Embedding_T<T>::Dim();

    // Decide which subtree to search first
    bool goLeft;
    if constexpr (std::is_same_v<T, float>) {
        goLeft = Node<T>::queryEmbedding < node->embedding;  // 1D case
    } else {
        goLeft = getCoordinate(Node<T>::queryEmbedding, axis) < getCoordinate(node->embedding, axis);  // Multi-D case
    }
    
    Node<T>* near = goLeft ? node->left : node->right;
    Node<T>* far = goLeft ? node->right : node->left;

    knnSearch(near, depth + 1, K, heap);

    // Check if we need to search the far subtree
    float splitDist;
    if constexpr (std::is_same_v<T, float>) {
        splitDist = std::abs(Node<T>::queryEmbedding - node->embedding);  // 1D case
    } else {
        splitDist = std::abs(getCoordinate(Node<T>::queryEmbedding, axis) - getCoordinate(node->embedding, axis));  // Multi-D case
    }
    
    if ((int)heap.size() < K || splitDist < heap.top().first) {
        knnSearch(far, depth + 1, K, heap);
    }
}