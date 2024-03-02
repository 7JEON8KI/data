# 사용할 Python 베이스 이미지 지정
FROM python:3.11

# 코드를 복사할 작업 디렉토리 생성
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt .

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 현재 디렉토리의 모든 파일을 작업 디렉토리로 복사
COPY . .

# 애플리케이션을 실행할 명령어 지정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# 컨테이너가 리스닝할 포트 번호 지정
EXPOSE 8000
