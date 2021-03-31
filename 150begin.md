# 150

<details>
    <summary>150.逆波兰表达式求值</summary>

根据 逆波兰表示法，求表达式的值。
有效的运算符包括 +, -, *, / 。每个运算对象可以是整数，也可以是另一个逆波兰表达式。

```javascript
var evalRPN = function(tokens) {
let operations = {"+":(a,b) => a+b,"-":(a,b) => a-b,'*':(a,b) => a*b,"/":(a,b) => Math.trunc(a/b)};
let stack = [];
for (let token of tokens){
    if (operations[token])
    {
        let number_2 = stack.pop();
        let number_1 = stack.pop();
        stack.push(operations[token](number_1, number_2));
    }
    else {
        stack.push(parseInt(token));
        }
}    
    return stack.pop();
};
```
</details>