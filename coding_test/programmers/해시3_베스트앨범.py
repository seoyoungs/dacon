'''
문제 설명
스트리밍 사이트에서 장르 별로 가장 많이 재생된 노래를 두 개씩 모아 베스트 앨범을 출시하려 합니다. 노래는 고유 번호로 구분하며, 노래를 수록하는 기준은 다음과 같습니다.

속한 노래가 많이 재생된 장르를 먼저 수록합니다.
장르 내에서 많이 재생된 노래를 먼저 수록합니다.
장르 내에서 재생 횟수가 같은 노래 중에서는 고유 번호가 낮은 노래를 먼저 수록합니다.
노래의 장르를 나타내는 문자열 배열 genres와 노래별 재생 횟수를 나타내는 정수 배열 plays가 주어질 때, 베스트 앨범에 들어갈 노래의 고유 번호를 순서대로 return 하도록 solution 함수를 완성하세요.

제한사항
genres[i]는 고유번호가 i인 노래의 장르입니다.
plays[i]는 고유번호가 i인 노래가 재생된 횟수입니다.
genres와 plays의 길이는 같으며, 이는 1 이상 10,000 이하입니다.
장르 종류는 100개 미만입니다.
장르에 속한 곡이 하나라면, 하나의 곡만 선택합니다.
모든 장르는 재생된 횟수가 다릅니다.
입출력 예
genres	plays	return
["classic", "pop", "classic", "classic", "pop"]	[500, 600, 150, 800, 2500]	[4, 1, 3, 0]
입출력 예 설명
classic 장르는 1,450회 재생되었으며, classic 노래는 다음과 같습니다.

고유 번호 3: 800회 재생
고유 번호 0: 500회 재생
고유 번호 2: 150회 재생
pop 장르는 3,100회 재생되었으며, pop 노래는 다음과 같습니다.

고유 번호 4: 2,500회 재생
고유 번호 1: 600회 재생
따라서 pop 장르의 [4, 1]번 노래를 먼저, classic 장르의 [3, 0]번 노래를 그다음에 수록합니다.
'''
def solution(genres, plays):      
    answer = []
    playsDict = {} # { 장르 : 총 재생 횟수 } 사전
    d = {} # { 장르 : [ ( 플레이 횟수, 고유 번호 ) ] }
    
    # 사전들 초기화
    for i in range(len(genres)):
        genre = genres[i]
        play = plays[i]
        playsDict[genre] = playsDict.get(genre,0) + play
        d[genre] = d.get(genre, []) + [(play, i)]
        
    # 재생 횟수 내림차순으로 장르별로 정렬
    genreSort = sorted(playsDict.items(), key=lambda x: x[1], reverse=True)
    
    # 재생 횟수 내림차순, 인덱스 오름차순 정렬
    # 장르별로 최대 2개 선택
    for (genre, totalPlay) in genreSort:
        d[genre] = sorted(d[genre], key=lambda x: (-x[0], x[1]))
        '''
        # 파이썬의 sorted 함수는 강력합니다. 
        # "key"라는 메소드 파라미터에 람다 함수를 넘겨주었는데, 
        # 다음처럼 2개의 키를 준것을 확인할 수 있습니다.
        # 여기서는 (플레이 횟수, 고유 번호) 형태로 리스트를 저장하기에,
        #  이 2개를 키로 넘겨줄 수 있는 것입니다. 참고적으로 -를 붙이면 내림차순,
        #  +를 붙이면 오름차순으로 정렬됩니다. 
        # 물론 이 때 키는 숫자형 데이터타입이어야 합니다.
        '''
        answer += [idx for (play, idx) in d[genre][:2]]
    
    return answer



