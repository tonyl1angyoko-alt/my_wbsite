# Lecture Intro3---template

先来看基本语法：

```cpp
1 #include <iostream>
 2 usingnamespace std;
 3
 4 template <typename T>
 5 TgetMax(Ta,Tb){
 6 return (a>=b)? a:b;
 7 }
 8
 9 intmain(){
 10 cout<<getMax(2,3)<<endl;
 11 cout<<getMax(2.0,5.3)<<endl;
 12 cout<<getMax<double>(3,4.2)<<endl;
 13 }
```

basically，用的时候不需要call数据？只是特别指出才需要

{% code fullWidth="false" %}
```cpp
 1 #ifndef LINK_SET_H_INCLUDED
 2 #define LINK_SET_H_INCLUDED
 3 #include "set.h"
 4
 5 template <typename elemType>
 6 class linkSet: public set<elemType>{
 7 struct node{
 8 elemTypeval;
 9 node*next;
 10 node(elemTypev,node*n)
 11 :val(v),next(n){}
 12 };
 13
 14 node*first;
 15 public:
 16 linkSet(){first=nullptr;}
 17 virtual ~linkSet();
 18 virtual bool add(elemTypeval);
 19 virtual bool find(elemTypeval) const;
 20 };
 22 template <typename elemType>
 23 linkSet<elemType>::~linkSet(){
 24 node*tmp;
 25 while(first){
 26 tmp=first;
 27 first=first->next;
 28 delete tmp;
 29 }
 30 }
 31
 32 template <typename elemType>
 33 boollinkSet<elemType>::add(elemTypeval){
 34 node*tmp;
 35 if(find(val)) returnfalse;
 36 tmp= newnode(val,first);
 37 first=tmp;
 38 return true;
 39 }
```
{% endcode %}

### Q：为什么22行要重新定义一次模板？

因为template定义域有些许奇怪。在第 5 行的 `template <typename elemType>` 是用来修饰紧随其后的 `class linkSet` 声明的。这种紧凑的性质可以增强代码鲁棒性。

###
