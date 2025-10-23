---
description: 第2章 线性表
---

# Lecture 02---linear table

## 线性表的基础

### 线性表的基本运算: 构建、属性、遍历类&#xD;

构建类运算\
**创建一个空的线性表**\
**清除线性表**\
`virtual void clear() = 0;`\
这是一个抽象基类 (Abstract Base Class, ABC)，它定义了一个“线性表”的接口（Interface）。

* 什么是接口？ 它只规定一个“线性表”必须能做什么（比如`clear`, `length`），但完全不管它具体怎么做（是用数组还是用链表）。
* 关键点： 注意所有函数都以 `virtual ... = 0;` 结尾。
  * `virtual` 关键字是实现多态的基础，它告诉编译器：这个函数的最终实现要到子类（派生类）里去找。
  * `= 0` 是C++的精髓之一，它声明这是一个纯虚函数 (Pure Virtual Function)。
  * 重难点： 任何包含纯虚函数的类（比如你这个`linearList`类）都是抽象类。你不能直接创建它的实例（比如 `linearList<int> myList;` 会编译失败），你只能创建它“具体子类”的实例（比如 `sequentialList<int> myList;` 或 `linkedList<int> myList;`，前提是这些子类实现了所有纯虚函数）。

这套设计，就是强迫所有“自称”是`linearList`的子类，都必须提供这一整套标准服务

属性类运算\
**获取表长𝑛**\
`virtual int length() const = 0;`

`const` (重难点)： 这是本行最重要的关键字。

* 它承诺：调用 `length()` 函数 绝不会修改 线性表对象内部的任何数据。它是一个只读 (read-only) 操作。
* 为什么重要？ 这允许你对一个 `const`（常量）线性表对象调用 `length()`。如果你不加 `const`，编译器会禁止 `const` 对象调用它，因为编译器担心这个函数会“偷偷”修改数据。
* 实现： 在子类中，它可能只是 `return currentLength;` 或 `return size;`

\
**访问元素**: 获取线性表中𝑖,(0≤𝑖≤𝑛−1)位置上的数据元素的值\
`virtual elemType visit(int i) const = 0;`\


遍历类运算\
**按位置从小到大顺序对线性表中每个元素访问且仅访问一次**\
`virtual void traverse(void (*touch)(const elemType &x)) const = 0;`

* 缺点： 如果 `elemType` 是一个很大的对象（比如一个复杂的 `struct` 或 `string`），创建副本的开销会很大。
* 对比： 另一种常见设计是 `virtual const elemType& visit(int i) const = 0;` （返回常量引用），这样可以避免复制，提高效率，同时 `const` 保证了调用者也不能修改它。



### 线性表的基本运算: 数据操纵类

插入元素𝑥使其成为𝑖,(0≤𝑖≤𝑛)位置上的数据元素\
`virtual void insert(int i, const elemType &x) = 0;`

删除𝑖,(0 ≤𝑖 ≤𝑛−1)位置上的数据元素\
`virtual void remove(int i) = 0;`

将𝑖,(0 ≤ 𝑖 ≤𝑛−1)位置上的数据元素的值修改成𝑥\
`virtual void update(int i, const elemType &x) = 0;`

从位置𝑖,(0≤𝑖 ≤𝑛−1)开始查找数据元素𝑥\
`virtual int search(const elemType &x, int i = 0) const = 0;`

```
#ifndef LIST_H_INCLUDED
#define LIST_H_INCLUDED

template <class elemType>
class list
{
public:
    class listError{};
public:
    virtual ~list(){}
    virtual void clear()=0;
    virtual int length() const=0;
    virtual elemType visit(int i) const=0;
    virtual void traverse(void(*touch)(const elemType &x)) const=0;
    virtual void insert(int i, const elemType &x) =0;
    virtual void remove(int i)=0;
    virtual void update(int i, const elemType &x) =0;
    virtual int search(const elemType &x, int i = 0) const =0;
};

#endif //LIST_H_INCLUDED
```

**异常类 (Exception Class)**

* `class listError{};` (第8行)
  * 分析：这是一个“标记类” (Marker Class)，用于异常处理。
  * 眼前一亮：这个类本身是空的，但它的类型就是它的信息。当子类（比如 `visit` 或 `remove`）发现一个错误（如索引越界），它就可以 `throw listError();` 来抛出一个异常。
  * 调用者可以通过 `catch(const list<elemType>::listError& e)` 来捕获这种特定于 `list` 的错误，从而做出相应处理。这是一种简单但有效的异常机制。

**析构函数 (Destructor)**

* `virtual ~list(){}` (第10行)
  * 分析：这是本页PPT中重难点之一：虚析构函数 (Virtual Destructor)。
  *   为什么必须？ 因为这个类是用来实现多态的。你很可能会写出这样的代码：

      C++

      ```
      // 基类指针指向一个子类对象
      list<int>* myList = new sequentialList<int>(); 
      // ... 使用 myList ...

      // 关键在这里！
      delete myList; 
      ```
  * 重难点：如果没有 `virtual` 关键字，`delete myList;` 只会调用基类 `list` 的析构函数。子类 `sequentialList` 用来释放内存（比如 `delete[] data`）的析构函数将永远不会被调用，导致灾难性的内存泄漏 (Memory Leak)！
  * 结论：加上 `virtual`，C++会确保在 `delete` 基类指针时，先正确调用子类的析构函数，再调用基类的。这里的空`{}`实现是说：基类自己没什么要清理的，但它为所有子类“打开了”正确析构的大门。

**在调用的时候，这样给出**

`#include "list.h"`\


`template <class elemType>`&#x20;

\
`class seqList : public list<elemType>`

只有继承的子类可以实例化



### 顺序表基本运算的实现-构建类

```cpp
template<typename elemType>
seqList<elemType>::seqList(int initSize){
    if (0 >= initSize)
        initSize = 8;

    elemArray = new elemType[initSize];
    maxSize = initSize;
    elemNum = 0;
}

template<typename elemType>
seqList<elemType>::~seqList(){
    delete []elemArray;
}

template<typename elemType>
void seqList<elemType>::clear(){
    elemNum = 0;
}
```
