#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>


using Embedding_T = float;

float distance(const Embedding_T &a, const Embedding_T &b);

// extract the scalar itself
constexpr float getCoordinate(Embedding_T e, size_t axis);

// KD-tree node
struct Node
{
    Embedding_T embedding;
    // std::string url;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static Embedding_T queryEmbedding;
};

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
Node *buildKD(std::vector<std::pair<Embedding_T, int>> &items, int depth = 0);

/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;



void freeTree(Node *node);

/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<PQItem, std::vector<PQItem>, std::less<PQItem>>;


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
 void knnSearch(Node *node,
               int depth,
               int K,
               MaxHeap &heap);