
n  = 1260
count = 0

# 큰 단위부터 작은 단위로
coin = [500, 100, 50, 10]

for i in coin:
    count += n // i # 해당 화폐로 거슬러 줄 수 있는 동전 세기(나머지 출력)
    n %= i

print(count) # 최소 동전의 개수




