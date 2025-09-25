# Lecture Intro2---virtual

## 析构函数为什么通常为虚函数

示例：

```cpp
 1 #include <iostream>
 2 usingnamespace std;
 3
 4 class base1{
 5 int *basePnt;
 6 public:
 7 base1(){basePnt= new int{0};}
 8 ~base1(){ delete basePnt;}
 9 virtual void who(){
 10 cout<< "base1"<<endl;
 11 }
 12 };
 13 class derive1: public base1{
 14 int *derivePnt;
 15 public:
 16 derive1(){derivePnt= newint{0};}
 17 ~derive1(){ deletederivePnt;}
 18 void who(){
 19 cout<< "derive1"<<endl;
 20 }
 21 };
 23 class base2{
 24 int *basePnt;
 25 public:
 26 base2(){basePnt= newint{0};}
 27 virtual ~base2(){ delete basePnt;}
 28 };
 29 class derive2: public base2{
 30 int *derivePnt;
 31 public:
 32 derive2(){derivePnt= new int{0};}
 33 ~derive2(){ deletederivePnt;}
 34 };
 35
 36 intmain(){
 37 base1*pobj1= newderive1;
 38 base2*pobj2= newderive2;
 39 pobj1->who();
 40 pobj1->base1::who();
 41 delete pobj1;
 42 delete pobj2;
 43 return 0;
 44 }

```

### 总结：

`37 base1 *pobj1 = new derive1;`

* 这里发生了 多态。我们创建了一个 `derive1` 类的对象，但是用一个 基类 `base1` 的指针 `$pobj1$` 来指向它。
* 在编译时，编译器只知道 `$pobj1$` 是一个 `base1` 类型的指针。但在运行时，程序知道它实际指向的是一个 `derive1` 对象

`39 pobj1->who();`

* 因为 `who()` 在 `base1` 中是 `virtual` 的，所以这里会发生 动态绑定。
* 程序在运行时检查 `$pobj1$` 实际指向的对象类型，发现是 `derive1`，因此调用 `derive1::who()`。
* 输出: `derive1`

`40 pobj1->base1::who();`

* 这里使用了作用域解析运算符 `::` 来强制调用 `base1` 类中定义的 `who()` 函数版本。
* 这种调用方式是 静态绑定 的，它忽略了多态性。
* 输出: `base1`

`41 delete pobj1;` (核心问题点)

* 我们通过一个 基类指针 (`$base1*$` ) 来删除一个 派生类对象。
* 编译器检查 `$pobj1$` 的类型，是 `base1*`。然后它去看 `base1` 的析构函数 `~base1()` 是否为 `virtual`。
* 它发现 `~base1()` 不是 `virtual` 的。
* 因此，编译器执行 静态绑定。它只知道要调用指针类型对应的析构函数，即 `~base1()`。
* 后果：
  1. 只有 `~base1()` 被调用，`$basePnt$` 指向的内存被释放。
  2. `~derive1()` 完全没有被调用！
  3. `derivePnt` 指向的内存没有被释放，造成了 内存泄漏！

## 我想知道base1\* pobj1 = new derive1;这句话的本质是什么？和int \*x = 有什么区别？

`int *x = new int;` 的本质是 纯粹的、一对一的内存管理。

* 你申请了一块“整数”大小的内存。
* 你用一个“指向整数”的指针来持有这块内存的地址。
* 指针的类型 (`int*`) 和它指向的内存中数据的类型 (`int`) 完全匹配。
* 这里没有任何“花哨”的行为，就是一个直接的“指针指向一块特定类型的数据”。

#### `base1* pobj1 = new derive1;` 的本质：抽象与多态

现在我们来分析这个复杂得多的例子：

1. `base1 *pobj1`：
   * 这声明了一个名为 `pobj1` 的变量。
   * 它的类型是 `base1*`，意思是“一个指向 `base1` 类对象的指针”。
   * 这是指针的 静态类型 (Static Type) 或 编译时类型。在编译器眼中，`pobj1` 就是一个指向 `base1` 的指针。
2. `new derive1`：
   * `new` 同样向堆内存申请一块空间。
   * 但这次的大小是 `sizeof(derive1)` 字节。因为 `derive1` 继承自 `base1` 并有自己的成员，所以 `sizeof(derive1)` 通常 大于 `sizeof(base1)`。
   * 关键区别：`new` 不仅仅是分配内存。它还会**调用构造函数**来初始化这块内存。调用顺序是：首先调用基类 `base1` 的构造函数，然后调用派生类 `derive1` 的构造函数，最终形成一个完整的 `derive1` 对象。
   * `new` 返回这个被完整构造好的 `derive1` 对象的起始地址。
3. `=`：
   * 赋值操作，把一个 `derive1` 对象的地址，存放到一个 `base1*` 类型的指针 `pobj1` 中。

为什么这个赋值是合法的？ 因为 `derive1` 公有继承自 `base1`，它们之间存在一种 "is-a"（是一个） 的关系。C++认为“一个 `derive1` 对象也是一个 `base1` 对象”（因为它包含了 `base1` 的所有成员和功能）。所以，用一个基类指针指向一个派生类对象是类型安全的，也是完全合法的。

`base1* pobj1 = new derive1;` 的本质是 抽象、向上转型 (Upcasting) 与多态。

* 类型不匹配：指针的静态类型 (`base1*`) 和它指向的对象的实际类型——即 动态类型 (Dynamic Type) (`derive1`) —— 不一致。这正是多态的基础。
* 视角限制：虽然 `pobj1` 指向一个完整的 `derive1` 对象，但通过 `pobj1` 这个指针，你只能访问到 `derive1` 对象中从 `base1` 继承来的那一部分成员和函数。你无法通过 `pobj1`   直接访问 `derive1` 自己独有的成员（比如 `derivePnt`）。指针的静态类型决定了你的“访问权限”或“视角”。
* 行为的动态性：这个不匹配的组合，为 `virtual` 关键字创造了舞台。当你通过 `pobj1` 调用一个虚函数（如 `who()`）时，程序会忽略指针的静态类型 (`base1*`)，而去查看指针所指向对象的动态类型 (`derive1`)，从而调用 `derive1::who()`。这就是动态绑定。

### 析构函数为什么通常为虚函数：

1. 派生类的析构函数会自动调用基类的析构函数。
2. 使用基类指针调用非虚函数: 调用基类的函数。
3. 使用基类指针调用虚函数: 调用指针所指实际对象类对应的函数。
4. 将析构函数设置为虚函数可避免多态调用中内存泄露的发生。
