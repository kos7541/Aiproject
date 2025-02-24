# migrate_2019_whitepaper.py

# 필요한 라이브러리 불러오기
# pandas: 데이터프레임 처리 및 CSV 파일 로드
# sentence_transformers: 텍스트 임베딩 생성 (자연어 처리)
# pymilvus: Milvus 벡터 데이터베이스 관리
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility


# 1️⃣ CSV 파일 로드
def load_csv(file_path):
    """
    주어진 경로에서 CSV 파일을 로드하고 데이터프레임으로 반환하는 함수.
    - cp949 인코딩을 사용해 한글 깨짐 문제 방지.
    - 로드한 데이터의 기본적인 확인 메시지를 출력.

    Args:
        file_path (str): CSV 파일의 경로

    Returns:
        DataFrame: 로드된 데이터프레임
    """
    print("📥 CSV 파일 불러오는 중...")
    data = pd.read_csv(file_path, encoding='EUC_KR')  # 한글 인코딩 적용
    print("✅ CSV 파일 불러오기 완료!")
    return data


# 2️⃣ 텍스트 데이터를 임베딩 벡터로 변환
def generate_embeddings(data, text_column):
    """
    데이터프레임의 특정 텍스트 컬럼을 임베딩 벡터로 변환하는 함수.
    - 사전 학습된 'all-MiniLM-L6-v2' 모델 사용
    - 임베딩 벡터는 384차원 벡터로 생성됨

    Args:
        data (DataFrame): 임베딩 생성 대상 데이터프레임
        text_column (str): 임베딩 생성에 사용할 텍스트 컬럼명

    Returns:
        DataFrame: 임베딩 컬럼이 추가된 데이터프레임
    """
    print("🔢 임베딩 생성 중...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 사전 학습된 BERT 기반 모델 로드
    data['embedding'] = data[text_column].apply(lambda x: model.encode(str(x)).tolist())  # 텍스트 임베딩 생성
    print("✅ 임베딩 생성 완료!")
    return data


# 3️⃣ Milvus 연결 및 컬렉션 생성
def create_milvus_collection(collection_name):
    """
    Milvus 데이터베이스에 연결하고, 주어진 이름의 컬렉션을 생성.
    - 기존 컬렉션이 존재하는 경우 삭제 후 재생성
    - 컬렉션 스키마 정의 (Primary Key 및 필드 설정)

    Args:
        collection_name (str): 생성할 컬렉션의 이름

    Returns:
        Collection: 생성된 Milvus 컬렉션 객체
    """
    print("🔗 Milvus 서버에 연결 중...")
    connections.connect(host='localhost', port='19530')  # Milvus 서버 연결

    # 기존 컬렉션 삭제 (이미 존재할 경우)
    if utility.has_collection(collection_name):
        print(f"🗑️ 기존 컬렉션 '{collection_name}' 삭제 중...")
        utility.drop_collection(collection_name)
        print(f"✅ 기존 컬렉션 '{collection_name}' 삭제 완료!")

    # 새로운 컬렉션 스키마 정의
    print("🏗️ 컬렉션 생성 중...")
    fields = [
        FieldSchema(name="IDX_PX", dtype=DataType.INT64, is_primary=True, auto_id=False),  # Primary Key 설정
        FieldSchema(name="IDX_NM", dtype=DataType.VARCHAR, max_length=20000),  # 인덱스 이름 (문자열)
        FieldSchema(name="WP_HTML_TRSF_CN", dtype=DataType.VARCHAR, max_length=20000),  # 텍스트 데이터
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # 384차원 임베딩 벡터
    ]
    schema = CollectionSchema(fields, description="2019 백서 데이터")  # 컬렉션 스키마 설정
    collection = Collection(name=collection_name, schema=schema)  # 컬렉션 생성
    print("✅ 컬렉션 생성 완료!")
    return collection


# 4️⃣ 데이터 삽입
def insert_data_to_milvus(collection, data):
    """
    Milvus 컬렉션에 데이터 삽입
    - 데이터프레임에서 필요한 컬럼 추출
    - 컬렉션에 삽입

    Args:
        collection (Collection): 데이터가 삽입될 Milvus 컬렉션 객체
        data (DataFrame): 삽입할 데이터프레임
    """
    print("📤 데이터 삽입 중...")
    data['WP_HTML_TRSF_CN'] = data['WP_HTML_TRSF_CN'].apply(str)
    idpx = data['IDX_PX'].tolist()  # Primary Key 값 추출
    idxNm = data['IDX_NM'].tolist()  # 인덱스 이름 데이터 추출
    trsfCN = data['WP_HTML_TRSF_CN'].tolist()  # 텍스트 데이터 추출
    embeddings = data['embedding'].tolist()  # 임베딩 데이터 추출

    collection.insert([idpx, idxNm, trsfCN, embeddings])  # 데이터 삽입
    collection.flush()  # 삽입된 데이터를 커밋
    print("✅ 데이터 삽입 완료!")


# 5️⃣ 인덱스 생성
def create_index(collection):
    """
    Milvus 컬렉션에 임베딩 데이터 기반 인덱스 생성
    - L2 거리 기반의 IVF_FLAT 인덱스 생성
    - 인덱스를 통해 빠른 유사도 검색 가능

    Args:
        collection (Collection): 인덱스를 생성할 Milvus 컬렉션 객체
    """
    print("⚡ 인덱스 생성 중...")
    index_params = {
        "metric_type": "L2",  # 거리 기반 측정 (L2 norm 사용)
        "index_type": "IVF_FLAT",  # 인덱스 알고리즘 설정
        "params": {"nlist": 128}  # 인덱스 파라미터 설정 (클러스터 개수)
    }
    collection.create_index(field_name="embedding", index_params=index_params)  # 인덱스 생성
    print("✅ 인덱스 생성 완료!")


# 6️⃣ 데이터 검색 테스트
def search_similar_data(collection, data):
    """
      Milvus 컬렉션에서 유사한 데이터를 검색하는 함수.
      - 컬렉션을 메모리에 로드한 후 검색
      - 첫 번째 임베딩 벡터를 기준으로 3개의 유사한 데이터를 검색

      Args:
          collection (Collection): 검색할 Milvus 컬렉션 객체
          data (DataFrame): 검색할 임베딩 데이터를 포함한 데이터프레임
      """
    print("🔍 데이터 검색 테스트 중...")

    # 1️⃣ 컬렉션을 메모리에 로드
    print("🚀 컬렉션 로드 중...")
    collection.load()  # 컬렉션 로드
    print("✅ 컬렉션 로드 완료!")

    # 2️⃣ 검색 파라미터 설정
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}  # L2 거리 기반 검색 설정
    embeddings = data['embedding'].tolist()  # 데이터프레임에서 임베딩 리스트로 변환

    # 3️⃣ 유사도 검색 수행 (첫 번째 임베딩 벡터 기준으로 3개의 유사 데이터 검색)
    print("🔍 임베딩 기반 검색 실행 중...")
    results = collection.search(embeddings[:1], "embedding", search_params, limit=3)

    # 4️⃣ 검색 결과 출력
    print("🎯 검색 결과:")
    for result in results[0]:  # 첫 번째 검색 결과에서 유사한 데이터 리스트
        print(f"ID: {result.id}, Distance: {result.distance}")

    print("✅ 검색 테스트 완료!")


# 7️⃣ 메인 실행 함수
def main():
    """
    전체 데이터 처리 및 Milvus 컬렉션에 데이터 삽입 및 검색 실행
    - CSV 파일 불러오기
    - 임베딩 생성
    - 컬렉션 생성 및 데이터 삽입
    - 인덱스 생성 및 검색 테스트
    """
    file_path = "C:\\milvus\\2019백서.csv"  # CSV 파일 경로 설정
    collection_name = "my_collection"  # 컬렉션 이름 설정

    # 단계별 실행
    data = load_csv(file_path)  # CSV 데이터 로드
    data = generate_embeddings(data, text_column='WP_HTML_TRSF_CN')  # 임베딩 생성

    collection = create_milvus_collection(collection_name)  # 컬렉션 생성
    insert_data_to_milvus(collection, data)  # 데이터 삽입
    create_index(collection)  # 인덱스 생성
    search_similar_data(collection, data)  # 검색 테스트 실행

    # Milvus 컬렉션 목록 출력
    collections = utility.list_collections()
    print(f"📊 현재 저장된 컬렉션 목록: {collections}")

    # Milvus에 저장된 데이터 개수 출력
    print(f"📈 저장된 데이터 개수: {collection.num_entities}")


# 스크립트 실행 (직접 실행 시 실행되도록 설정)
if __name__ == "__main__":
    main()