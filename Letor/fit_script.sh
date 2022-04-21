train_file=${1}
val_file=${2}
feature_file=${3}
metric=${4}
ranker=${5}
model_file=${6}
train_perf_file=${7}
time_file=${8}

start_time=$(date +%s.%3N)
java -jar ./RankLib/RankLib-2.17.jar -train "${train_file}" \
    -validate "${val_file}" \
    -feature "${feature_file}" \
    -metric2t "${metric}" \
    -ranker "${ranker}" \
    -save "${model_file}" >${train_perf_file}
end_time=$(date +%s.%3N)

echo "start_time,end_time" >${time_file}
echo "${start_time},${end_time}" >>${time_file}
