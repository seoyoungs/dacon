# 어떠한 수가 N이 1이 될 때 까지 다음의 두 과정 중 하나를 반복적으로 선택하여 수행하려고 한다. 
# 예시 N = 17, K =4 라고 가정하자



n, k = map(int, input().split())
result = 0

# N이 K 이상이라면 K로 계속 나누기
while n >= k:
    # N이 k로 나눠 떨어지지 않는다면 N에서 1씩 빼기
    while n % k !=0:
        n -= 1
        result += 1
    # k로 나누기
    n //= k
    result += 1

# 마지막으로 남은 수에 대해 1씩 빼기
while n > 1:
    n -= 1
    result += 1

print(result)



