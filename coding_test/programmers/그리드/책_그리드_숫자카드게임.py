# 숫자카드 게임
# 여러 개의 숫자카드 중에서 가장 높은 숫자가 쓰인 카드 한 장을 뽑는 게임이다.
'''
# 그 전에 2~9 단까지 모든 결과 출력하는 반복문 출력
for i in range(2, 10): # 2단 부터 9단 까지 출력
    for j in range(1, 10): # 1~9까지 구구단에 곱할 것
        print(i, "x", j, '=', i*j)
    print(i,'단')
    print()
'''

n,m = list(map(int, input().split()))

result=0
# 한 줄씩 받아서 확인
for i in range(n):
    data = list(map(int, input().split()))
    # 현재 줄에서 '가장 작은 수 찾기'
    min_value = min(data)
    # '가장 작은 수'중에서 '가장 큰 수'찾기
    result = max(result, min_value)

print(result)










