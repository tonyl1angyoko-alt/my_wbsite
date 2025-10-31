---
description: chap06树
---

# Lecture 03---树

### 结点数与边数的关系

<figure><img src="../../.gitbook/assets/image (1).png" alt=""><figcaption></figcaption></figure>

d叉树也称为位置树，需要提供固定的插槽，让子树插进去，不论是否存在可视为空。

### 二叉树不同形态的数目

<figure><img src="../../.gitbook/assets/image (2).png" alt=""><figcaption></figcaption></figure>

要构建一棵有 $$ $n$ $$ 个结点的二叉树，我们是这样思考的：

1. 从 $$ $n$ $$ 个结点中，我们先选出 1 个结点作为根 (Root)。
2. 现在我们还剩下 $$ $n-1$ $$ 个结点，需要分配给左子树和右子树。
3. 我们可以枚举所有可能的分配情况

所以和出栈序列有一样的结果。

### 一些定义

对一棵二叉树:\
(1) 若所有非叶结点的度为 2, 则称之为严格二叉树 (strict/full binary tree)\
(2) 若所有非叶结点的度为 2, 且所有叶结点具有相同的层次, 则称之为满二叉树或\
完美二叉树 (perfect binary tree)\
(3) 若其是 “将一棵完美二叉树的最后一层结点从右至左依次删掉若干个” 后形成\
的, 则称之为完全二叉树 (complete binary tree)

### 二叉链表的类定义

```cpp
#include "linkStack.h"
#include "linkQueue.h"

using namespace std;

template
class linkBinaryTree : public binaryTree {

public:
    enum treeType {perfect, complete, normal};

private:
    struct node {
        elemType data;
        node *left, *right;

        node() : left(nullptr), right(nullptr) {}
        node(const elemType &x, node *l = nullptr, node *r = nullptr)
            : data(x), left(l), right(r) {}
        ~node() {}
    };

    node *root;

    // Private helper functions
    void deleteTree(node *&r) const;
    int sizeType(treeType &type, const node *r) const;
    int height(const node *r) const;
    void levelOrder(const node *r, void (*touch)(const elemType &e)) const;
    void preOrder(const node *r, void (*touch)(const elemType &e)) const;
    void inOrder(const node *r, void (*touch)(const elemType &e)) const;
    void postOrder(const node *r, void (*touch)(const elemType &e)) const;

public:
    // Constructor & Destructor
    linkBinaryTree() : root(nullptr) {}
    ~linkBinaryTree() { deleteTree(root); }

    // Basic operations
    int sizeType(treeType &type) const { return sizeType(type, root); }
    virtual int size() const { treeType type; return sizeType(type); }
    virtual int height() const { return height(root); }

    // Traversal operations
    virtual void levelOrder(void (*touch)(const elemType &e)) const {
        return levelOrder(root, touch); 
    }
    virtual void preOrder(void (*touch)(const elemType &e)) const {
        return preOrder(root, touch); 
    }
    virtual void inOrder(void (*touch)(const elemType &e)) const {
        return inOrder(root, touch); 
    }
    virtual void postOrder(void (*touch)(const elemType &e)) const {
        return postOrder(root, touch); 
    }

public:
    // Tree creation functions
    bool inputTree(const elemType &nil);

    bool createTreeFromInLevelOrder(
        const elemType inOrderArray[],
        const elemType levelOrderArray[],
        int nodesNum);

    bool createTreeFromInPreOrder(
        const elemType inOrderArray[],
        const elemType preOrderArray[],
        int nodesNum);

    bool createTreeFromInPostOrder(
        const elemType inOrderArray[],
        const elemType postOrderArray[],
        int nodesNum
```
