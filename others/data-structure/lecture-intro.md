---
description: the first class, about some tricky and confusing programming skills for C++
---

# Lecture Intro

titleconst 用法辨析

```cpp
 #include <iostream>
 2 struct demoType{
 3 int array[100];
 4 intgetValBI(intidx){ return array[idx];}
 5 intgetValGI(intidx) const {returnarray[idx];}
 6 };
 7 intgetValBO(demoType&obj, int idx){
 8 return obj.array[idx];
 9 }
 10 intgetValGO(const demoType&obj, intidx){
 11 return obj.array[idx];
 12 }
 13 intmain(){
 14 int i=3,j=4;
 15 const intk=5;
 16 const int*pi=&i;
 17 int *const qi=&i;
 18 int &ri=k;
 19 const int&rj=j;
 20 const int&rk=k;
 21 demoTypevarObj= {{1}};
 22 const demoTypeconObj={{2}};
 23 i=1;
 24 k=2;
 25 *pi=2;
 26 pi=&j;
 27 *qi=2;
 28 qi=&j;
 29 j=conObj.getValBI(0);
 30 j=conObj.getValGI(0);
 31 j=varObj.getValBI(0);
 32 j=varObj.getValGI(0);
 33 j=getValBO(varObj,0);
 34 j=getValGO(varObj,0);
 35 j=getValBO(conObj,0);
 36 j=getValGO(conObj,0);
 37 return 0;
 38 }

```

16行：你不能通过指针 pi 来修改它所指向的变量的值 (即 `*pi = ...` 是非法的)。但是，你可以让 `pi` 指向另一个地址 (即 `pi = ...` 是合法的)。我们称之为 "底层 const"。

17行：指针 `qi` 本身是常量，它必须在定义时初始化，并且之后不能再指向其他地址 (即 `qi = ...` 是非法的)。但是，你可以通过它来修改所指向变量的值 (即 `*qi = ...` 是合法的)。我们称之为 "顶层 const"

19行：定义了一个对 `const int` 的引用 `rj`，并绑定到普通变量 `j`。这意味着你不能通过引用 `rj` 来修改 `j` 的值，尽管 `j` 本身是可以被修改的

20行：指针 `rk` 不能再指向别的地址，同时也不能通过 `rk` 来修改它所指向的值

#### 总结代码中的实现错误如下：

* 第 23 行: `i = 1;`
  * 正确。`i` 是普通变量，可以赋值。
* 第 24 行: `k = 2;`
  * 错误。`k` 在第 15 行被声明为 `const` 常量，其值不能被修改。
* 第 25 行: `*pi = 2;`
  * 错误。`pi` 在第 16 行被声明为 `const int *`，即指向常量的指针。不能通过 `pi` 修改所指向地址 (`i`) 的值。
* 第 26 行: `pi = &j;`
  * 正确。`pi` 本身不是 `const` 的，所以可以改变它指向的地址。
* 第 27 行: `*qi = 2;`
  * 正确。`qi` 在第 17 行被声明为 `int * const`，它是一个常量指针，但它指向的 `int` 不是常量。因此可以通过 `qi` 修改 `i` 的值。执行后 `i` 的值变为 2。
* 第 28 行: `qi = &j;`
  * 错误。`qi` 在第 17 行被声明为常量指针，它不能再指向其他地址。
* 第 29 行: `j = conObj.getValBI(0);`
  * 错误。`conObj` 是一个 `const` 对象。`const` 对象只能调用 `const` 成员函数 (即函数声明末尾有 `const` 的函数)。`getValBI` 不是一个 `const` 成员函数，调用它可能会有修改对象成员的风险，所以编译器禁止这种行为。
* 第 30 行: `j = conObj.getValGI(0);`
  * 正确。`getValGI` 是一个 `const` 成员函数，可以被 `const` 对象 `conObj` 调用。
* 第 31 行: `j = varObj.getValBI(0);`
  * 正确。`varObj` 是一个普通对象，它可以调用任何成员函数 (const 或非 const)。
* 第 32 行: `j = varObj.getValGI(0);`
  * 正确。普通对象也可以调用 `const` 成员函数。
* 第 33 行: `j = getValBO(varObj, 0);`
  * 正确。将普通对象 `varObj` 传递给一个按值接收参数的函数。函数内部会创建一个 `varObj` 的副本。
* 第 34 行: `j = getValGO(varObj, 0);`
  * 正确。`getValGO` 函数接收一个 `const` 引用。将一个普通对象 `varObj` 传递给一个 `const` 引用是允许的，这是一种权限缩小，是安全的。
* 第 35 行: `j = getValBO(conObj, 0);`
  * 正确。将 `const` 对象 `conObj` 传递给一个按值接收参数的函数。函数内部会创建一个 `conObj` 的副本，这个副本本身不再是 `const` 的。
* 第 36 行: `j = getValGO(conObj, 0);`
  * 正确。将 `const` 对象 `conObj` 传递给一个接收 `const` 引用的函数，类型完美匹配。
* 第 37-38 行: `return 0;` 和 `}`
  * 程序正常结束。

## 面向编g程oc

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

[^1]: title
