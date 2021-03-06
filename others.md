## 程序员代码面试指南

### 1可见山峰对的数量

​		一个不含有负数的数组可以代表一圈环形山，每个位置的值代表山的高度。有两个方向：next方向(逆时针方向)，last方向(顺时针)。

山峰A和山峰B相互看见的条件为：

1. 如果A和B是同一座山，认为不能相互看见
2. 如果A和B是不同的山，并且在环中相邻，认为可以相互看见
3. 如果A和B是不同的山，并且在环中不相邻，假设两座山峰高度的最小值为min。如果A通过next或last方向到B的途中没有高度比min大的山峰，则认为A和B可以相互看见。

给定一个不含有负数且没有重复值的数组arr，请返回有多少对山峰能够相互看见。

**进阶问题**：给定一个不含有负数但可能含有重复值的数组arr，返回有多少对山峰能够互相看见

### 思路

<font color='red'>首先结果与数组内的数如何排列不会有关系。</font>

<font color='red'>不要管相邻不相邻，都是陷阱。</font>

采用由小找大的思路，假设山峰k的高度不是最大值和次大值，那么他在环状山中向自身左右2个方向找，一定能分别找到一个高度大于他自身的山峰H，找到的山峰H和山峰k自身组成一对（相邻也不相邻没关系）。山峰H后面的山不管怎么样都不再满足与山峰k可见的条件（山峰k被山峰H挡住了），山峰H前的山峰L是可能与山峰k可见的，但是这些都比山峰k的高度低，考虑这里由大到小的山峰对会在最后结果中重复，因此也不用管。所以一个这个的山峰K能产生2个可见山峰对。显然不是最高和第二高的山峰k总共有n-2个，能有2*(n-2)个山峰对，另外第二高的山能找到最高的组成一对，最高的找不到更高的了。所以最后结果是2*(n-2)+1	..。。。

<hr>

进阶：数组中有重复的数。

借助栈，仍然小找大，从最大值的地方开始遍历一圈回到最大值时结束。栈中存储的是（value，times），遍历时，遍历数组中值与栈顶元素相同则times+1，比栈顶value小则入栈，比栈顶value大，则<b>栈顶元组</b>出栈。出栈时计算山峰对。

遍历阶段有元素出栈，说明出栈元素在两边都有大于他的值。由上面思路的推论，考虑的找到比他大的，增加2*times个可见山峰对。考虑自身高度组成的对。既times个中任2个组合C(2,times)。特别的，times=1时，没有2个自身高度组成可见山峰对，C(2,1)=0.既res+=C(2,times)+2*times

遍历阶段结束后，栈中肯定还有元素。开始清算出栈。分三种情况：1，出栈元素不是栈低也不是倒数第二个。此时出栈增加的可见山峰对计算方法跟遍历阶段相同，因为他也能在两边找到比他大的元素。2，出栈元素倒数第二个，这个时候就看最后一个元素的times是否大于1，大于1则计算方法仍然同上，理由同上。等于1的情况下res+=C(2,times)+times，只能找到一个比他大的元素，所以不乘2了。 3，最后一个元素出栈，直接res+=C(2,times)，找不到比他更大的元素，只算内部组成。

over。

## 环状数组最大子序和

leetcode. 918题：给定一个由整数数组 `A` 表示的**环形数组 `C`**，求 `**C**` 的非空子数组的最大可能和。

```python
class Solution:
    def maxSubarraySumCircular(self, A: List[int]) -> int:
        res = 0
        maxcur=mincur=0
        maxsum=minsum=A[0]
        for x in A:
            res+=x
            maxcur = max(maxcur,0)+x
            maxsum = max(maxsum,maxcur)
            mincur = min(mincur,0)+x
            minsum = min(minsum,mincur)
        if res==minsum:  # 全为负数的情况
            return maxsum
        return max(maxsum,res-minsum)

```

解析：53题最大子序和的进阶版。数组为环状。以53题解法为基础

注意2点：<font color='red'>不全为负数的情况下，数组总和—最大子序和=最小子序和。</font>

<font color='red'>环形数组C中最大子序和，要么等于数组A中最大子序，要么包含A的首和尾，此时最大=总和-最小子序和</font>

##  最长上升子序列

300.最长上升子序列

给定一个无序的整数数组，找到其中最长上升子序列的长度。

解法一，动态规划， 复杂度On<sup>2</sup>，dp[ i ]表示以nums[i]结尾的最长上升子序列长度

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

```

解法二，（贪心）。非常巧妙。设置dp数组。dp的长度表示当前最长的上升子序列，dp[i]代表长度为i的序列末尾最小的数。dp中的数一定是一个升序，这样就可以用二分法。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp=[-float('INF')]  # 关键
        n=0
        for x in nums:
            if x > dp[-1]:
                dp.append(x)
                n+=1
            else:  # 此处可以转化为二分查找，找到刚好比x小的数。
                for i in range(n-1,-1,-1):
                    if dp[i]<x:
                        dp[i+1]=x
                        break
        return len(dp)-1
```

# 计算器

227实现加减乘除的计算器,无括号,无负数

```python
class Solution:
    def calculate(self, s: str) -> int:
        num = 0
        stack = list()
        op = '+'
        for i, c in enumerate(s):
            if c.isnumeric():
                num = num*10+int(c)
            if c in '+-*/' or i == len(s)-1:
                if op == '+':
                    stack.append(num)
                if op == '-':
                    stack.append(-num)
                if op == '*':
                    stack.append(stack.pop()*num)
                if op == '/':
                    stack.append(int(stack.pop()/num))
                op = c
                num = 0
        return sum(stack) 

```

224 ,基本计算器

加减乘除,有括号,无负数

```JavaScript
var calculate = function(s) {
    var i = -1;//全局
    var helper = function(){
        let q = [], n = '', f = '+'
        while(i++ < s.length-1 || n) {
            if (s[i] === ' ') continue
            if(s[i]==='(') {//递归算括号里面的
                n=helper()
                continue
                }
            if (/\D/.test(s[i])) {
                switch (f) {
                    case '+':
                        q.push(n) 
                    break;
                    case '-':
                        q.push(-n) 
                    break;  
                    case '*':
                        q.push(q.pop() * n) 
                    break;
                    case '/':
                        q.push(q.pop() / n | 0) 
                }
                if(s[i]===')') {
                    return q.reduce((p, v) => p + (v | 0), 0)
                }
                f = s[i], n = ''
            } 
            else n += s[i]; 
        }return q.reduce((p, v) => p + (v | 0), 0)
    }
    return helper()
};
```

中缀表达式转后缀表达式通用的方法。

1）如果遇到数，我们就直接将其加入到后缀表达式。

2）如果遇到左括号，则我们将其放入到栈中。

3）如果遇到一个右括号，则将栈元素弹出，将弹出的操作符加入到后缀表达式直到遇到左括号为止，接着将左括号弹出，但不加入到结果中。

4）如果遇到其他的操作符，如（“+”， “-”）等，从栈中弹出元素将其加入到后缀表达式，直到栈顶的元素优先级比当前的优先级低 （或者遇到左括号或者栈为空）为止。弹出完这些元素后，将当前遇到的操作符压入到栈中。

5）如果我们读到了输入的末尾，则将栈中所有元素依次弹出。

## 约瑟夫环问题

推公式f（n，m） = （f（n-1，m）+m）%n，f（n，m）表示最后剩余的数在当前队列中的索引

反推情况，每次队列的第一个往后移动m个位置，在前一个位置便是删掉的位置。反推n-1次，过程中最后生还者的索引一直在变，找到最后生还者的初始索引，此时索引正好对应本身数值

```js
var lastRemaining = function(n, m) {
var start=0;
for(let rest =2;rest<=n;rest++) start=(start+m)%rest;
return start;
};
```

