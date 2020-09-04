# leet code

欢迎阅读

## 目录
1. [经典算法类](#classic)  
	动态规划、
2. [经典数据结构](#data)  
	链表、栈、树、图
3. [数学](#math)  
	位运算、找规律
4. [特别技巧](#skill)  
	花式技巧
---
## <span id ='classic'>经典算法</span>

### DP动态规划

<details>
<summary>CLICK ME</summary>

<pre>summary标签与正文间一定要空一行！！！</pre>
</details>

---
## <span id="data">经典数据结构</span>
### 链表
### 栈
### 树
### 图
---
## <span id ='math'>数学类</span>
### 位运算
### 规律
---
## <span id ='skill'>技巧类</span>
<details>
<summary>1.两数之和</summary>
题目：找到数组中两数之和等于target，返回两数的索引
解法：1暴力法遍历，两数的所有组合，复杂度O(N<sub>2</sub>); 2使用哈希表（字典）保存已访问过的数和索引，时间复杂度O(N),空间复杂度O(N).
```
class Solution:
    def twoSum(self, nums, target):
        m = {}
        for k, v in enumerate(nums):
            if target - v in m.keys():
                return[m[target - v], k]
            m[v] = k
```
</details>


>>> 请问 Markdwon 怎么用？ - 小白

>> 自己看教程！ - 愤青

> 教程在哪？ - 小白
* 无序列表项 一
+ 无序列表项 二
- 无序列表项 三

使用 Markdown[^1]可以效率的书写文档, 直接转换成 HTML[^2]。

[^1]:Markdown是一种纯文本标记语言

[^2]:HyperText Markup Language 超文本标记语言
Markdown语法参考
# 一级标题
## 二级标题
##### 五级标题
- 列表第一项
- 列表第二项
1. 有序列表第一项
2. 有序列表第二项
[标题](链接地址)
![图片描述](图片链接地址)
*斜体*
**粗体**
> 引用段落
```代码块```
