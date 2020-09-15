# leet code 1-50

欢迎阅读

## 目录
1. [经典算法类](#classic)  
	动态规划、贪心、回溯
2. [经典数据结构](#data)  
	链表、栈、树、图
3. [数学](#math)  
	位运算、找规律
4. [特别技巧](#skill)  
	花式技巧
5. [其他](#other)    
   运算符优先级:~取反>算术运算符>移位>关系>按位与>逻辑与>条件运算>赋值
---
## <span id ='classic'>经典算法</span>

### DP动态规划
<details>
<summary>5.最长回文子串</summary>

题目：返回字符串中最长回文子串
解法一：动态规划。```dp[i][j]=1```表示i到j之间为回文。```dp[i,i]=1,dp[i][i+1]=(str[i]==str[i+1])```既自身单个算回文，自身与下一个相同算回文。状态转移方程```dp[i][j]=(dp[i-1][j+1] and str[i]==str[j])```。时间复杂度n^2,太慢了

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        ans = ""
        for l in range(n):  # 枚举子串的长度 l,从0开始
            for i in range(n):  # 枚举子串的起始位置 i
                j = i + l  # 结束位置
                if j >= n:
                    break
                if l == 0:
                    dp[i][j] = True
                elif l == 1:
                    dp[i][j] = (s[i] == s[j])
                else:
                    dp[i][j] = (dp[i + 1][j - 1] and s[i] == s[j])
                if dp[i][j] and j - i + 1 > len(ans):
                    ans = s[i:j+1]
        return ans
```
解法二：简接滑动窗口。运用切片和反向,分别对奇数偶数长度的子串判断。时间复杂度n(据说).

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:return ""
        maxLen,start=1,0
        for i in range(len(s)):  # i为结尾索引，
        	if i-maxLen >=1 and s[i-maxLen-1:i+1]==s[i-maxLen-1:i+1][::-1]:
        		start=i-maxLen-1
        		maxLen+=2
        		continue
        	if i-maxLen >=0 and s[i-maxLen:i+1]==s[i-maxLen:i+1][::-1]:
        		start=i-maxLen
        		maxLen+=1
        return s[start:start+maxLen]

```
解法三：中心扩展。先向后纳入相同元素,再分别2端扩张。时间复杂度n(不是).

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return s
        i = 0
        mx = 0
        left=0
        right=0
        while i < len(s):
            l = i - 1
            while i + 1 < len(s) and s[i + 1] == s[i]:
                i += 1
            r = i + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            if mx < r - l - 1:
                mx = r-l-1
                left, right=l+1, r-1
            i += 1
        return s[left:right+1]

```
</details>
### 贪心
<details>
<summary>12.整数转罗马数字</summary>

题目：罗马数字包含以下七种字符： I， V， X， L，C，D 和 M，给定一个整数，将其转为罗马数字。输入确保在 1 到 3999 的范围内
解法：按照1000，900，500，400，100，90，50，40，10，9，5，4，1。贪婪匹配，这个数字元素设计非常合理，确保了贪心的解一定是正确 的。
```python
class Solution:
    def intToRoman(self, num: int) -> str:
        strs = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]        
        ret = ""        
        for i, j in enumerate(nums):
            while num >= j:
                ret += strs[i]
                num -= j
            if num == 0:
                return ret
```
</details>
### 回溯
<details>
<summary>39.组合总数</summary>

题目：给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。数组中无重复的数，且每个数使用次数不限。
解法：回溯加剪枝，将数组candidates先排序，依次选择数来与target相减，更新下一层被选数组中的所有数都大于等于上一层被选的数，来达到剪枝效果。特别的，python中path参数更新，变相完成了回溯的效果。
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        solutions=[]
        def backtracking(cans, remain, path):
            for i, can in enumerate(cans):
                if remain-can == 0:
                    solutions.append(path+[can])
                elif remain-can > 0:
                    backtracking(cans[i:], remain-can, path+[can])
                else:
                    break
        candidates.sort()
        backtracking(candidates, target, [])
        return solutions
```
</details>
<details>
<summary>46.全排列，47全排列有重复</summary>

题目：46给定一个 没有重复 数字的序列，返回其所有可能的全排列。
解法：回溯经典。复杂度n*N！
```python
class Solution:
    def permute(self, nums):
        def backtrack(start, end):
            if start == end-1:
                ans.append(nums[:])
            for i in range(start, end):  # 每一层相当于固定start位的数，
                nums[start], nums[i] = nums[i], nums[start]  # 让start之后的数轮流固定
                backtrack(start+1, end)
                nums[start], nums[i] = nums[i], nums[start]
        ans = []
        backtrack(0, len(nums))
        return ans  
```
题目：47，给定一个有重复 数字的序列，返回其无重复的所有可能的全排列。
解法1：复杂度n*N！。因为存在重复元素，使用回溯时，要使用同一层不固定已出现过的数。
```python
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res=[]
        def backtrace(start,end):
            if start==end-1:
                res.append(nums[:])
            seen=set()
            for i in range(start,end):
                if i>start and nums[i] in seen:
                    continue
                seen.add(nums[i])
                nums[i],nums[start]=nums[start],nums[i]
                backtrace(start+1,end)
                nums[i],nums[start]=nums[start],nums[i]
        
        backtrace(0,len(nums))
        return res
```
解法2：不回溯，使用递归。必须得先排序。循环中，如果后面的数大于当前层要固定的位的数（对应代码中的索引i），则与之交换，注意此过程中i的值应该是一直增大的，因为之前递增排序，交换后索引i之后的数仍然有序，同时巧妙的避免了固定i位的数的重复。
```python
    def permuteUnique(self, nums):
        nums.sort()
        result = list()
        self._permuteunique(nums, 0, result)
        return result

    def _permuteunique(self, nums, i, result):
        if i == len(nums) - 1:
            result.append(nums.copy())
            return
        
        for k in range(i, len(nums)):
            if i != k and nums[i] == nums[k]:  # 避免重复的关键
                continue
            nums[i], nums[k] = nums[k], nums[i]
            self._permuteunique(nums.copy(), i + 1, result)
```
</details>
<details>
<summary>40.组合总数2</summary>

题目：给定一个全正整数元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。数组中有重复的数，且每个数只使用1次，且输出结果无重复。
解法：回溯加剪枝，同上，关键在于剪去同一层的相同元素的。同一层不选重复的数字
```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        candidates.sort()
        n = len(candidates)
        def helper(index, target ,path):
            if target == 0:
                res.append(path)
                return
            for i in range(index, n):
                if target < candidates[i]:
                    break
                if i > index and candidates[i - 1] == candidates[i]:  # 同一层剪枝
                    continue
                helper(i + 1 , target - candidates[i] , path + [candidates[i]] )
       
        helper(0 ,target,[])
        return res
```
</details>
---
## <span id="data">经典数据结构</span>
### 链表
<details>
<summary>2.两数相加</summary>

题目：两数逆序链表表示，求和的逆序链表表示
解法：按链表逐位相加，大于10则取个位，后面的和需+1。特别的，循环结束并不是2个指向链表的指针都为空时，还有进位。这题没啥意思，就是整数加法，正常朝左进位，逆序就朝右。
```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = p = ListNode(None)
        s = 0  # 进位用，
        while l1 or l2 or s:
            s +=(l1.val if l1 else 0) + (l2.val if l2 else 0)
            p.next = ListNode(s%10)
            p = p.next
            s  //= 10
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next
```
</details>
<details>
<summary>19.删除链表的倒数第N个节点</summary>

题目：删除链表的倒数第N个节点
解法1：经典快慢双指针。
```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dum = ListNode(0)      # 设置虚拟指针的目的是防止删除第一个节点
        dum.next = head
        cur = head
        pre = dum
        for i in range(n):     # 先走n步
            cur = cur.next
        while cur:             # 再走剩余的步，最后pre指向的就是要删除节点的前面一个节点
            cur = cur.next
            pre = pre.next
        pre.next = pre.next.next  # 删除这个节点
        return dum.next
```
解法2：递归骚操作。
```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        def index(node):
            if not node:
                return 0
            i = index(node.next) + 1
            if i > n:
                node.next.val = node.val
            return i
        index(head)
        return head.next
```
</details>
<details>
<summary>21.合并2个有序链表</summary>

题目：合并2个有序链表
解法：老经典题了，当年考研还考了合并2个升序数组。
```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        prev = prehead = ListNode(-1)
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next           
            prev = prev.next
        prev.next = l1 if l1 is not None else l2
        return prehead.next
        #  递归骚操作
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 and l2:
            if l1.val > l2.val: l1, l2 = l2, l1
            l1.next = self.mergeTwoLists(l1.next, l2)
        return l1 or l2
```
</details>
### 栈
<details>
<summary>20.有效的括号</summary>

题目：给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
解法：用栈.先放一个元素进去，谨防空栈出操作
```python
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False 
        return len(stack) == 1
```
</details>
<details>
<summary>23.两两交换链表中的节点</summary>

题目：第1个与第二交换，第三换第4，一次类推
解法：把2个要交换的节点找到，断开，链接一顿操作，。
```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        p=head
        res =pre = ListNode(0)
        pre.next = p
        while p and p.next:
            q=p.next
            p.next=q.next
            q.next=p
            pre.next=q
            pre=p
            p=p.next
        return res.next
```
</details>
### 树
### 图
---
## <span id ='math'>数学类</span>
### 位运算
<details>
<summary>29.两数相除</summary>

题目：两数相除不能用乘除法。可以挨个减法，但太慢，所以每次翻倍减
解法：可以挨个减法，但太慢，所以每次翻倍减。另负数越界问题，将被除数都弄成负数来解决
```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        b=abs(divisor)
        if dividend==-2147483648 and divisor==-1:
            return 2147483647
        a = -dividend if dividend>0 else dividend
        res=0
        while a+b<=0:
            count=0
            temp=b
            while a+temp<=0:
                count+=1
                temp+=temp
            res+=1<<count-1
            a+=temp>>1
        return -res if (dividend>0) ^ (divisor>0) else res
```
</details>
<details>
<summary>50.pow(x,n)</summary>
题目：实现 pow(x, n) ，即计算 x 的 n 次幂函数。
解法：分治法，总共要实现n个x相乘。将转化二进制数。比如77 的二进制表示 1001101，对应着77=1+4+8+64.即2<sup>0</sup>,2<sup>2</sup>,2<sup>3</sup>,2<sup>6</sup>.于是有，x<sup>77</sup>=x * x<sup>4</sup> * x<sup>8</sup> * x<sup>64</sup>.时间复杂度logN。

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        m = -n if n < 0 else n
        y = 1
        while m:
            if m & 1:
                y *= x
            x *= x
            m >>= 1
        return y if n >= 0 else 1/y
```
</details>
### 规律
<details>
<summary>6.Z字形变换</summary>

题目：将一个给定字符串根据给定的行数，以从上往下、从左到右进行 Z 字形排列。
解法一：找规律,字符串按索引又第一行排到第n行,再反向排到第1行,设置转向flag.
```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows < 2: return s
        res = [""]*numRows
        i, flag = 0, -1
        for c in s:
            res[i] += c
            if i == 0 or i == numRows - 1: flag = -flag
            i += flag
        return "".join(res)
```
解法二：找规律,寻找排列周期,T=2*numRows-2
```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1:return s
        res = ['']*numRows
        turn = numRows*2-2
        for i in range(len(s)):
            a =i % turn
            if a<numRows:
                res[a]+=s[i]   
            else:
                res[turn-a]+=s[i]       
        return ''.join(res)
```
</details>
<details>
<summary>38.外观数列</summary>

题目：「外观数列」是一个整数序列，从数字 1 开始，序列中的每一项都是对前一项的描述。前五项如下：

1.     1
2.     11
3.     21
4.     1211
5.     111221
解法：找生成新的字符串的规律，可以用双指针，也可以计数迭代
```python
class Solution:
    def countAndSay(self, n):
        if n<1: return None
        layer='1'
        for _ in range(1,n):
            count,target,newlayer=0,layer[0],''
            for char in layer:
                if char==target:
                    count+=1
                else:
                    newlayer += str(count)+target
                    count= 1
                    target =char
            newlayer += str(count)+target
            layer = newlayer
        return layer
```
</details>
<details>
<summary>48.旋转图像</summary>

题目：给定一个 n × n 的二维矩阵表示一个图像。将图像顺时针旋转 90 度。需原地操作
解法：1.先上下反转，再转置。(最舒服）2.先转置，再水平翻转。3.找出规律，索引[i][j]将移动到索引[j][n-i-1].  复杂度n^2
```python
class Solution:
    def rotate(self, A: List[List[int]]) -> None:  # 解法1
        A.reverse()
        for i in range(len(A)):
            for j in range(i):
                A[i][j], A[j][i] = A[j][i], A[i][j]
    def rotate(self, A: List[List[int]]) -> None:  # 解法3
        n = len(matrix[0])        
        for i in range(n // 2 + n % 2):
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp
```
</details>
---
## <span id ='skill'>技巧类</span>
<details>
<summary>1.两数之和</summary>

题目：找到数组中两数之和等于target，返回两数的索引
解法：1暴力法遍历，两数的所有组合，复杂度O(N<sup>2</sup>); 2使用哈希表（字典）保存已访问过的数和索引，时间复杂度O(N),空间复杂度O(N).
```python
class Solution:
    def twoSum(self, nums, target):
        m = {}
        for k, v in enumerate(nums):
            if target - v in m.keys():
                return[m[target - v], k]
            m[v] = k
```
</details>
<details>
<summary>16.三数之和</summary>

题目：找到数组中三数之和等于0，返回三数
解法：本题是两数之和进阶版，暴力发超时.其实这题也是双指针题，但是速度太慢。转为两数和，x2+x3 = target= -x1.需考虑特殊情况0，和重复数字。18题四数和不做了，
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        dic = defaultdict(int)   # 先用字典记录每个数字出现的次数
        for i in nums:
            dic[i] += 1
        nums = sorted(dic)     # 得到一个排序后的去重数组
        for i, x in enumerate(nums):   
            if x == 0 and dic[x] > 2:  # 情况一：x 为0，且 x 出现了3次及3次以上
                res.append([0, 0, 0])      
            if x != 0 and dic[x] > 1:    # 情况二：若 x（0除外）出现了2次及2次以上，
                if -2 * x in dic:    #  只要-2乘 x 在数组里有出现，便符合条件
                    res.append([x, x, -2*x])
            if x < 0:      #情况三：这里是剪枝效果，固定X<y<z,保证res里不重复
                y_z = -x          #y+z 的和为 -x 便能符合要求
                z_id = bisect.bisect_right(nums,y_z//2) # 求得 z最小可能的下标
                for z in nums[z_id:]:    # 则 z 的取值范围是 nums[z_id:]
                    y = y_z - z
                    if y > x and y in dic:
                        res.append([x, y, z])
        return res
```
</details>
<details>
<summary>3.无重复字符的最长字串</summary>

题目：找到字符串中无重复字符的最长字串
解法：此题经典滑动窗口问题。  
解法一，使用字典记录出现位置，出现相同字符时，i-start为一个合法的字串长度。注意相同字符上次出现的索引一定要不小于起始索引。
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        start, res, dic = 0, 0, {}
        for i, c in enumerate(s):
            if c in dic and dic[c] >= start:  # c上次出现的索引一定要不小于起始索引
                res = max(res, i-start)
                start = dic[c]+1  # 起始位置更新
            dic[c] = i   
        res = max(res, len(s)-start)
        return res
```
	解法二，使用双端队列deque，模拟滑动窗口，当遇到相同字符时，队列左侧排除元素，直到排出到该字符停，该方法非常直观，好理解，速度较解法一慢点。
```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        queue = collections.deque()
        if not s:return 0
        res = 1
        for s1 in s:
            if s1 in queue:
                res = max(res,len(queue))
                while True:
                    if queue.popleft() == s1:
                        break
            queue.append(s1)
        res = max(res,len(queue))
        return res
```
</details>
<details>
<summary>11.盛最多水的容器</summary>

题目：输入数组 [1,8,6,2,5,4,8,3,7]。数字代表挡板高度，容器能够容纳水的最大值为 49。
解法：1经典双指针题，双指针首尾开始向中间移动，保证不会错过最大面积，每次移动的指针为数值较小的那一个
```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        i, j = 0, len(height) - 1
        water = 0
        while i < j:
            water = max(water, (j - i) * min(height[i], height[j]))
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return water
```
</details>
<details>
<summary>13.最长公共前缀</summary>

题目：编写一个函数来查找字符串数组中的最长公共前缀。不存在公共前缀，返回 ""。
解法：1按规则循环比较，依次按索引i比较每个串的第i位是否相同。
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs: return ""
        cmn = ""
        for i,c in enumerate(strs[0]):
            for s in strs:
                if i>=len(s) or s[i] != c:
                    return cmn
            cmn+=c
        return cmn
```
解法2：骚操作，运用zip*。
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        out_str = ''
        for i in zip(*strs):
            if len(set(i)) == 1:
                out_str += i[0]
            else:
                break
        return out_str
```
</details>
<details>
<summary>49.字母异位词分组</summary>

题目：给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。
解法：利用质数相乘唯一性，可以将相同字母不同顺序的字符串映射到唯一的一个整数值。

然后很神奇的是，将字母c的值映射为ord(c)+2,也可以通过。
不要问为啥，我也不知道。
而且我测试过，映射为ord(c)+3，ord(c)+4，ord(c)+1，ord(c)+0都不行，会出现乘积重复val值
但是ord(c)+2就是可以。
```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        mp = {}
        for word in strs:
            val = 1
            for c in word:
                val *= (ord(c) + 2)
            if mp.get(val):
                mp[val].append(word)
            else:
                mp[val] = [word]
        return list(mp.values())
```
</details>
<details>
<summary>17.电话号码的字母组合</summary>

题目：给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。
解法：其实就是拼音9建，按数字能打出的所有字母可能性.组合问题，递归，回溯都可以。下面代码骚操作，使用itertools.product(*)
```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        b = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
        return [] if digits == "" else [ "".join(x) for x in itertools.product(*(b[d] for d in digits ))]
```
</details>
<details>
<summary>17.搜素旋转排序数组</summary>

题目：假设按照升序排序的数组在预先未知的某个点上进行了旋转。数组无重复
( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

解法：二分法，分四种情况。要用索引0当指标，因为数据可能一直递增。注意第一个判断里面等号不可少。
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        L, H = 0, len(nums)   # 注意h从len(nums)开始而不是len(nums)-1
        while L < H:
            M = (L+H) // 2
            if nums[M] < nums[0] <= target: # +inf
                H = M
            elif nums[M] > nums[0] > target: # -inf
                L = M+1
            elif nums[M] < target:
                L = M+1
            elif nums[M] > target:
                H = M
            else:
                return M
        return -1
```
</details>
<details>
<summary>81.搜素旋转排序数组2，有重复</summary>

题目：假设按照升序排序的数组在预先未知的某个点上进行了旋转。
( 例如，数组 [0,0,1,2,2,5,6] 可能变为 [2,5,6,0,0,1,2] )。
搜索一个给定的目标值，如果数组中存在这个目标值，则返回TRUE，否则返回 false 。
解法：二分法，17题的进阶。主要问题在于mid等于L时，不知道target在左还是右边。与17题相比较，增加mid等于L时，L+=1.同时判断在左右有序序列里时用的时nums[L]
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        L, H = 0, len(nums) 
        while L < H:
            M = (L+H) // 2
            if nums[M]==target:
                return True
            if nums[M]==nums[L]:  # 关键点
                L+=1
                continue
            if nums[M] < nums[L] <= target: # 变为nums[L]
                H = M
            elif nums[M] > nums[L] > target: # -inf
                L = M+1
            elif nums[M] < target:
                L = M+1
            else:
                H = M
        return False
```
</details>
<details>
<summary>34. 在排序数组中查找元素的第一个和最后一个位置</summary>

题目：给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
解法：还是二分法，还是分二次，找到mid == target之后，在用二分找第一和最后出现的位置
```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        l,r=0,len(nums)-1
        first=last=-1
        while l<=r:  # find target
            m=(l+r)//2
            if nums[m]<target:
                l=m+1
            elif nums[m]>target:
                r=m-1
            else:
                first=last=m
                while l<first:    #find first position
                    mid=(l+first)//2
                    if nums[mid]==target:
                        first=mid
                    else:
                        l=mid+1
                while last<r:   #find last position
                    mid=(r+last+1)//2  # 取后一个位置
                    if nums[mid]==target:
                        last=mid
                    else:
                        r=mid-1
                break
        return (first,last)
```
</details>
<details>
<summary>7.搜索插入的位置</summary>

题目：给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。数组中无重复元素。
解法：二分法
```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums)
        while start < end: 
            mid = (start + end) // 2
            if nums[mid] < target: 
                start = mid + 1
            else: 
                end = mid
        return start 
  # 调库函数
return bisect.bisect_left(nums,target)
```
</details>
<details>
<summary>43.字符串相乘</summary>

题目：给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
解法：此题其实没啥意思。不让把整个数转化为整数来计算，可以把单个字符转化为数字，为了出题而出题。1个想法是把num2中每一个数与num1相乘再求和，此过程会增加很多补位的0的加法运算。复杂度mn+n^2.mn为乘法次数，n^2为加法次数。  
第2个想法是用列表记录，两数长度分别为m，n。则两数乘积长度最大为m+n，至少为m+n-1.
num1中的第i个数和num2中第j个数相乘，所得结果对最后的结果中第i+j+1有累计。复杂度mn（不知道怎么算的？）不过确实减少了数字很大的数相加。
```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        if num1 == "0" or num2 == "0":
            return "0"
        
        m, n = len(num1), len(num2)
        ansArr = [0] * (m + n)
        for i in range(m - 1, -1, -1):
            x = int(num1[i])
            for j in range(n - 1, -1, -1):
                ansArr[i + j + 1] += x * int(num2[j])  # 关键点，对i+j+i索引的累加
        # 处理进位，最后位是个位，往前进位，可能会出现超过3位数，ansAri[i]超过100
        for i in range(m + n - 1, 0, -1):  
            ansArr[i - 1] += ansArr[i] // 10
            ansArr[i] %= 10
        
        index = 1 if ansArr[0] == 0 else 0
        ans = "".join(str(x) for x in ansArr[index:])
        return ans
```
</details>
## <span id ='other'>其他</span>
<details>
<summary>7.整数反转</summary>

题目：给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
解法：没什么好说 的.注意负数和整数越界,32 位的有符号整数其数值范围为 [−2^31,  2^31 − 1]
```python
class Solution:
    def reverse(self, x: int) -> int:
        sign = 1
        if x < 0 : sign = -1
        x = x * sign
        res = 0
        while x :
            res = res * 10 + x % 10
            x //= 10
        return res * sign if res < 2 ** 31 else 0
```
</details>
<details>
<summary>9.回文数</summary>
题目：判断一个整数是否是回文数
解法：方法1转化为字符串，```return str(num)==str(num)[::-1]```.  
	方法2，将整数反转，如上第7题  
	方法3，将整数反转一半。（方法4，循环将整数的首位与末位比较）

```python
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x==0:return True
        if x<0 or x%10==0:return False
        reversed = 0
        while x>reversed:
            reversed = reversed*10 + x % 10
            x//=10
        return x == reversed or x==reversed//10
```
</details>
<details>
<summary>13.罗马数字转整数</summary>

题目：字符串表示罗马字符，返回整数
解法：首先建立一个HashMap来映射符号和值，然后对字符串从左到右来，如果当前字符代表的值不小于其右边，就加上该值；否则就减去该值。以此类推到最左边的数，最终得到的结果即是答案。也可以反过迭代，比如下面
```python
class Solution:
    def romanToInt(self, s: str) -> int:
        dict = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
        total=dict[s[-1]]
        for i in range(2,len(s)+1):
            if dict[s[-i]]<dict[s[-i+1]]:
                total-=dict[s[-i]]
            else:
                total+=dict[s[-i]]
        return total
```
</details>
<details>
<summary>22.括号生成</summary>

题目：数字 n 代表生成括号的对数，生成所有可能的并且 有效的 括号组合
解法：DFS
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        valid = []
        def rec(l,r,s):
            if l+r == n*2:
                valid.append(s)
            if l < n:
                rec(l+1,r,s+'(')
            if r < l:
                rec(l,r+1,s+')')
        
        rec(0,0,"")
        return valid
```
</details>
<details>
<summary>31.下一个排列</summary>

题目：实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）
解法：从后向前找到i索引的值大于i-1，然后再从后向前找到第一个k索引的值大于i-1，交换i-1和k。最后将i之后的数都反转。
```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        n = len(nums)-1
        k = n
        for i in range(n,0,-1):
            if nums[i-1]<nums[i]:
                while nums[k]<=nums[i-1]:
                    k-=1
                nums[i-1],nums[k] = nums[k],nums[i-1]
                nums[i:]=sorted(nums[i:])
                break
        else:
            nums.reverse()
```
</details>
<details>
<summary>36.有效的数独</summary>

题目：判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。
数字 1-9 在每一行只能出现一次。
数字 1-9 在每一列只能出现一次。
数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。

解法：每行每列每个3X3盒子都设一个数组，数组的index+1对应数字index+1在该行（列）的出现次数（初始为0，出现一次加1，大于1时return false）。
```python
class Solution:
    def isValidSudoku(self, board):
        rows = [[0]*9 for _ in range(9)]
        columns = [[0]*9 for _ in range(9)]
        boxs = [[0]*9 for _ in range(9)]

        for i in range(9):
            for j in range(9):
                num = board[i][j]
                if num != '.':
                    num = int(num)
                    boxIndex = (i // 3) * 3 + j // 3
                    rows[i][num-1] += + 1
                    columns[j][num-1] += 1
                    boxs[boxIndex][num-1] += 1
                    if rows[i][num-1] > 1 or columns[j][num-1] > 1 or boxs[boxIndex][num-1] > 1:
                        return False
        return True
```
</details>
