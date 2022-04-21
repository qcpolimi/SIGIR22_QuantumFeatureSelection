test_file=${1}
model_file=${2}
base_test_performance_file=${3}

test_measures=("MAP"
	"NDCG@1" "NDCG@2" "NDCG@3" "NDCG@4" "NDCG@5"
	"NDCG@6" "NDCG@7" "NDCG@8" "NDCG@9" "NDCG@10"
	"P@1" "P@2" "P@3" "P@4" "P@5"
	"P@6" "P@7" "P@8" "P@9" "P@10")

for m in "${test_measures[@]}"; do
    test_performance_file="${base_test_performance_file}_${m}.txt"
    java -jar ./RankLib/RankLib-2.17.jar -load "${model_file}" -test "${test_file}" -metric2T "${m}" >${test_performance_file}
done