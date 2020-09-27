# leetcode 50-100



不分类了，按顺序整。

<details>
    <summary>53.最大子序和</summary>

题目：给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。
解法：一动态规划，dp[i]表示以索引i结尾的子数组的最大和，因必须连续，所以dp[i]要么为dp[i-1]+nums[i],要么等于nums[i]自身。状态转移方程dp[i]=max(dp[i-1]+nums[i],nums[i]).可以时间n，空间1

二贪心，实则与方法一区别不大，下面代码中原地操作，遍历过程中如果上一个数大于0变相加。取数组最大值即可，越等于将nums数组当作dp数组。此法已经非常妙了

```python
class Solution:  # 方法二
    def maxSubArray(self, nums: List[int]) -> int:
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
        return max(nums)
class Solution:  # 方法三
    def maxSubArray(self, nums: List[int]) -> int:
        def help(l,r):
            if l==r:
                return (nums[l],nums[l],nums[l],nums[l])
            else:
                m = (l+r)>>1
                left = help(l,m)
                right = help(m+1,r)
                lsum = max(left[0],left[-1]+right[0])
                rsum = max(right[1],left[1]+right[-1])
                mxun = max(left[2],right[2],left[1]+right[0])
                isum = left[-1]+right[-1]
                return (lsum,rsum,mxun,isum)      
        res=help(0,len(nums)-1)
        return res[2]
```
三分治法，实现复杂，生成线段树后可以logn的时间实现数组任意长度内的最大子序和求解
</details>
<details>
    <summary>54.螺旋矩阵与59.螺旋矩阵2</summary>

题目：54给定一个包含 m x n 个元素的矩阵（m 行, n 列），按照顺时针螺旋顺序，返回矩阵中的所有元素。
解法：1分层剥离，根据顺时针遍历索引的规律。2每次排除第一层，然后逆时针旋转90度。重复。

```python
class Solution:  # 方法1
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return list()
        
        rows, columns = len(matrix), len(matrix[0])
        order = list()
        left, right, top, bottom = 0, columns - 1, 0, rows - 1
        while left <= right and top <= bottom:
            for column in range(left, right + 1):
                order.append(matrix[top][column])
            for row in range(top + 1, bottom + 1):
                order.append(matrix[row][right])
            if left < right and top < bottom:
                for column in range(right - 1, left, -1):
                    order.append(matrix[bottom][column])
                for row in range(bottom, top, -1):
                    order.append(matrix[row][left])
            left, right, top, bottom = left + 1, right - 1, top + 1, bottom - 1
        return order
class Solution:  # 方法二
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        return matrix and [*matrix.pop(0)] + self.spiralOrder([*zip(*matrix)][::-1])
```
题目：59给定一个正整数 n，生成一个包含 1 到 n^2 所有元素，且元素按顺时针顺序螺旋排列的正方形矩阵。
解法：1直接顺时针生成，2从一个数开始，每次顺时针旋转90°，然后生成一行，长度为已生成矩阵的行数，补在第一行。
```python
class Solution:
    def generateMatrix(self, n: int) -> [[int]]:
        l, r, t, b = 0, n - 1, 0, n - 1
        mat = [[0 for _ in range(n)] for _ in range(n)]
        num, tar = 1, n * n
        while num <= tar:
            for i in range(l, r + 1): # left to right
                mat[t][i] = num
                num += 1
            t += 1
            for i in range(t, b + 1): # top to bottom
                mat[i][r] = num
                num += 1
            r -= 1
            for i in range(r, l - 1, -1): # right to left
                mat[b][i] = num
                num += 1
            b -= 1
            for i in range(b, t - 1, -1): # bottom to top
                mat[i][l] = num
                num += 1
            l += 1
        return mat
class Solution:    # 方法二
    def generateMatrix(self, n: int) -> List[List[int]]:
        A, lo = [[n*n]], n*n
        while lo > 1:
            lo, hi = lo - len(A), lo
            A = [list(range(lo, hi))] + list(zip(*A[::-1]))
        return A
```
</details>

<details>
    <summary>54.螺旋矩阵与59.螺旋矩阵2</summary>

题目：给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个位置。
解：方法1，从前面开始跳。设置ma为最大可达到索引位置。遍历数组，每次更新最远位置，如果出现索引i大于ma，返回False。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n=len(nums)
        ma = 0
        for i in range(n):
            if i>ma:
                return False
            ma=max(nums[i]+i,ma)
        return True         
```
方法2，从后往前跳。设置目标值target初始为末尾，向前遍历时，如果i + nums[i]>=target，说明当前索引i可以到达目标，更新目标位置为新的target，最后判断target==0。等于说明能从0索引到底最后。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        target = len(nums) - 1
        for i in range(len(nums) - 1, -1, -1):
            if i + nums[i] >= target:
                target = i
        return target == 0                     
```
</details>
<details>
    <summary>56.合并区间</summary>

题目：给出一个区间的集合，请合并所有重叠的区间。
解：主要是需要排序，排序后前一段的终点大于后一段的起点贼合并。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort()
        pre = intervals[0]
        res =[]
        for s in intervals:
            if pre[1]>=s[0]:
                pre[1]=max(s[1],pre[1])
            else:
                res.append(pre)
                pre = s
        res.append(pre)
        return res                     
```
</details>
<details>
    <summary>58.最后一个单词的长度</summary>

题目：给定一个仅包含大小写字母和空格 ' ' 的字符串 s，返回其最后一个单词的长度。如果字符串从左向右滚动显示，那么最后一个单词就是最后出现的单词。。
解：无。
```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        result = 0
        for word in s[::-1]:
            if word.isalpha():
                result += 1
            elif result != 0:
                return result
        return result                   
```
</details>

<details>
    <summary>60.第k个排序</summary>

题目：给出集合 [1,2,3,…,n]，其所有元素共有 n! 种排列。返回第k大小的排列。
解：固定排列第一位，剩下n-1个数，有（n-1）！种排列，得出数学规律。

```python
class Solution:
    def getPermutation(self, n: int, k: int) -> str:
        nums=list(range(1,1+n))
        res=''
        base = math.factorial(n)
        while nums:
            base=base//n
            index=(k-1)//base  # 注意要k-1，因为是第k。
            res+=str(nums.pop(index))
            k=k%base
            n-=1
        return res                
```
</details>
<details>
    <summary>61.旋转链表</summary>

题目：给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。
解：计算链表长度，然后将首尾相接，向前移动 k%lengthe断开。

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head or k==0:
            return head
        count = 1
        p = head
        while p.next:
            count+=1
            p=p.next
        p.next=head
        k=count-k%count
        while k:
            p=p.next
            k-=1
        res=p.next
        p.next=None
        return res                        
```
</details>
<details>
    <summary>62.不同路径。63.不同路径2。64最小路径和</summary>

题目：62,一个机器人位于一个 m x n 网格的左上角 。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。问总共有多少条不同的路径？。
解：方法一，经典动态规划，状态转移方程dp[i][j] = dp[i-1][j]+dp[i][j-1]。时间空间复杂度：O(N^2)
方法一可以优化到空间复杂度On。
方法二，求数学组合， return int(comb(m+n-2,n-1))，向下走m-1步，向右走n-1步。求个组合
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        f = [[1]*n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i-1][j] + f[i][j-1]
        return f[m-1][n-1]
class Solution:  # 动态规划优化
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1]*n  # 第一行全为1，只有一种方式
        for i in range(1, m):
            for j in range(1, n):
                dp[j] = dp[j] + dp[j-1]
        return dp[-1]  
```
题目：63，网格中存在障碍物，输入为矩阵，网格中的障碍物和空位置分别用 1 和 0 来表示。
解法：多一次判断，如果遇到障碍，将到达该位置的方式置为0.此时存在不能达到的位置了，因此不能再初始化为1了，仅仅将入口出初始化为1.既dp[0]=1.
```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        n,m = len(obstacleGrid[0]), len(obstacleGrid)
        dp = [1] + [0]*n  # 多一位置0，用于j索引等于0时，j-1为-1的问题
        for i in range(m):
            for j in range(n):
                dp[j] = 0 if obstacleGrid[i][j] else dp[j]+dp[j-1]
        return dp[-2]  
```
题目：64，给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。。
解法：与62类似，动态规划，状态转移方程改变一下：dp[i][j] = (dp[i-1][j],dp[i][j-1])+grid[i][j].既到达一个位置得最小距离等于他上面的位置与左边位置中选一个较小的路径，再加上自身。特殊考虑第一行和第一列即可。
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        n=len(grid[0])
        m=len(grid)
        dp = [0]*n
        for i in range(m):
            for j in range(n):  
                if (i and j):
                    dp[j]=min(dp[j]+grid[i][j],grid[i][j]+dp[j-1])
                elif j==0:
                    dp[j]+=grid[i][j]
                else:
                    dp[j]=dp[j-1]+grid[i][j]
        return dp[-1]                
```
</details>
<details>
    <summary>66.加1</summary>

题目：给定一个由整数组成的非空数组所表示的非负整数，在该数的基础上加一。
最高位数字存放在数组的首位， 数组中每个元素只存储单个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

解：无。
```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        n=len(digits)-1
        while digits[n]==9:
            digits[n]=0
            n-=1
            if n<0:
                return [1]+digits
        digits[n]+=1
        return digits                  
```
</details>
<details>
    <summary>67.二进制求和</summary>

题目：给你两个二进制字符串，返回它们的和（用二进制表示）。
输入为 非空 字符串且只包含数字 1 和 0。。

解：1遍历字符串，从个位开始相加，考虑进位。2内置函数直接return '{:b}'.format(int(a, 2) + int(b, 2))。3，算是题外解。利用位运算，不用加减乘除。如何实现真正的二进制加法。
```python
class Solution:
    def addBinary(self, a, b) -> str:
        x, y = int(a, 2), int(b, 2)
        while y:
            answer = x ^ y  # 第一步不考虑进位情况下两数相加的结果
            carry = (x & y) << 1  # 需要进位的地方，左移一位正好等于进位
            x, y = answer, carry  # 重复，第一步的结果加上进位的值
        return bin(x)[2:]       
```
</details>
## 还没做完
