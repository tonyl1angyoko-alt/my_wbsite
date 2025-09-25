---
description: the first class, about some tricky and confusing programming skills for C++
---

# Lecture Intro1---OOP

## 面向过程编程

<pre class="language-cpp"><code class="lang-cpp">#ifndef INT_SET_H_INCLUDED
 2
 3 typedef struct intSet_t{
 4 int *elemArray;
 5 int maxSize;
 6 int elemNum;
 7 }intSet;
 8
 9 intSet*createIntSet(int initSize);
 10 voiddestroyIntSet(intSet**ppSet);
 11 intaddElem(intSet*pSet, intval);
 12 intfindElem(const intSet*pSet, intval);
 13
 14 #endif /*#ifndef INT_SET_H_INCLUDED */
<a data-footnote-ref href="#user-content-fn-1"> code/chap01/intSetC/main.c</a>
 1 #include &#x3C;stdio.h>
 2 #include "intSet.h"
 3
 4 int main(){
 5 intSet*pSet;
 6 intval,ret;
 7
 8 pSet=createIntSet(8);
 9 addElem(pSet,3);
 10 addElem(pSet,5);
 11 addElem(pSet,2);
 12 val=4;
 13 ret=findElem(pSet,val);
 14 printf("%d%sin theset.\n",
 15 val,
 16 (ret? "is" :"isn't"));
 17 destroyIntSet(&#x26;pSet);
 18
 19 return 0;
 20 }

</code></pre>

## 解读如下：

参数类型是 `intSet **ppSet`，一个指向指针的指针（双重指针）。这是为了在函数内部不仅能释放集合占用的内存，还能将调用者（例如 `main` 函数）中的指针变量（如 `pSet`）设置为 `NULL`，防止其成为“悬垂指针”（dangling pointer）

* 单指针 `intSet *p` 作为参数：函数得到一个地址，可以修改 这个地址指向的数据。
* 双重指针 `intSet **pp` 作为参数：函数得到一个指针的地址，它不仅可以修改这个指针指向的数据，更可以修改 这个指针本身（让它指向别处，或者指向 `NULL`）。

{% hint style="info" %}
C++值传递。为了修改本身的内容需要传进来地址。为了修改地址的内容（例如变成null）需要传进来地址的地址。
{% endhint %}

`#include "intSet.h"`：包含我们自己定义的整数集合的头文件。使用双引号表示在当前项目目录中查找该文件。

这段代码清晰地展示了C++语言中“接口与实现分离”的核心思想。`intSet.h` 定义了“做什么”（what），而 `main.c` 则展示了“如何使用”（how），具体的实现（`createIntSet` 等函数的代码体）则隐藏在另一个未展示的 `.c` 文件中。代码风格良好，特别是指针的使用（尤其是 `const` 和双重指针）体现了作者对内存管理和程序健壮性的深入考虑

### malloc 补课

1. 包含头文件
   * 必须包含 `#include <stdlib.h>`。
2. 分配内存
   * 使用 `sizeof` 计算所需字节数，并通过强制类型转换将 `void*` 结果赋给你的指针。
   * `int *arr = (int*) malloc(10 * sizeof(int));`
3. 检查是否成功
   * 必须 检查 `malloc` 是否返回 `NULL`，以处理内存不足的情况。
   * `if (arr == NULL) { /* 错误处理 */ }`
4. 释放内存
   * 使用完毕后，必须用 `free()` 将内存归还给系统，否则会造成内存泄漏。
   * `free(arr);`
   * 良好习惯：释放后将指针设为 `NULL`，防止误用。`arr = NULL;`

关键原理：`malloc` 返回的是所分配内存块的 起始地址。你可以通过指针算术（例如使用 `arr[i]` 的数组语法）来访问这块内存中的任何位置。程序员需要手动管理这块内存的整个生命周期，从申请、使用到最终释放。In one word, to make it convinient.

## 面向对象编程

```cpp
 1 #ifndef INT_SET_H_INCLUDED
 2
 3 class intSet{
 4 //private://default access attribute
 5 int *elemArray;
 6 int maxSize;
 7 int elemNum;
 8
 9 public:
 10 intSet(intinitSize=8);
 11 ~intSet();
 12
 13 bool addElem(intval);
 14 bool findElem(intval) const;
 15 };
 16
 17 #endif //#ifndef INT_SET_H_INCLUDED
 code/chap01/intSetCpp/main.cpp
 1 #include <iostream>
 2 #include "intSet.h"
 3 using namespace std;
 4
 5 int main(){
 6 intSetset;
 7 int val;
 8 bool ret;
 9
 10 set.addElem(3);
 11 set.addElem(5);
 12 set.addElem(2);
 13 val=4;
 14 ret=set.findElem(val);
 15 cout<<val
 16 <<(ret? "is ":"isn't")
 17 <<"in theset."<<endl;
 18
 19 return 0;
 20 }
```

#### 为什么这就是OOP了？

面向对象编程（Object-Oriented Programming）的核心思想是将 数据 和 操作数据的函数 捆绑在一起，形成一个独立的对象。这段C++代码完美地体现了OOP的几个核心特性：

1. 封装 (Encapsulation)
   * C语言：数据 (`struct intSet`) 和操作 (`create`, `addElem` 等函数) 是分离的。任何拿到 `intSet*` 指针的代码原则上都可以直接修改其内部成员（如 `pSet->elemNum`），这破坏了数据的完整性。
   * C++：数据 (`elemArray` 等) 和操作 (`addElem` 等)被封装在 `intSet` 类这个统一的单元中。数据被设为 `private`，只能通过 `public` 的成员函数来访问和修改。这就像给数据上了一把锁，并只提供几个安全的钥匙（公有函数）来操作它，极大地提高了代码的安全性和可维护性。
2. 抽象 (Abstraction)
   * 使用者（`main` 函数）只需要知道 `intSet` 对象能做什么（通过 `addElem`, `findElem` 等公有接口），而完全不需要关心其内部是如何用数组实现的。类的实现细节被隐藏了起来。
3. 构造函数与析构函数 (Automated Lifecycle Management)
   * C语言：程序员必须手动、正确地调用 `createIntSet` 和 `destroyIntSet`。忘记调用 `destroyIntSet` 会导致内存泄漏；忘记调用 `createIntSet` 而直接使用指针会导致程序崩溃。
   * C++：对象的创建和销毁是自动管理的。只要你声明一个对象 `intSet set;`，构造函数就确保它被正确初始化。当对象离开作用域时，析构函数确保它被正确清理。这种将资源管理与对象生命周期绑定的技术称为 RAII (Resource Acquisition Is Initialization)，是C++最强大的特性之一，它极大地简化了资源管理并减少了错误。

[^1]: title
