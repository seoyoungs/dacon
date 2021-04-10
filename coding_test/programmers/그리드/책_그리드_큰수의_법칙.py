
# n = [2,4,5,4,6]
# m = 8
# k=3
# N,M,K를 공백으로 구분하여 입력하기
n,m,k = map(int, input().split())
# N개의 수를 공백으로 구분하여 입력하기
data = list(map(int, input().split()))

data.sort() # 입력받은 데이터 정렬하기
first = data[n-1] # 가장큰수
second = data[n-2] # 두번째로 큰 수

# 가장 큰 수가 더해지는 횟수 계산
count = int(m / (k + 1)) * k
count += m % (k + 1)

result = 0
result += (count) * first # 가장 큰 수 더하기
result += (m - count) * second # 두 번째로 큰 수 더하기

print(result) # 최종 답안 출력






