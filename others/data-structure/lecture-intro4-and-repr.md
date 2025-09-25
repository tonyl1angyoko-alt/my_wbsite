# Lecture Intro4---\*\&repr

## const 用法辨析

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
