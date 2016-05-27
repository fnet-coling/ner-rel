raw_data=../../Data/BBN/
data=../../Intermediate/BBN/
output=../../Results/
dim=50
lr=0.01
max_iter=20
thread=7
delimter='/'
max_dep=3
./warp $data $dim $lr $max_iter $thread $output
python python_scripts/create_embedding_file.py $output/warp_B.txt $data/type.txt $output/warp_B.bin
python python_scripts/create_embedding_file.py $output/warp_A.txt $data/feature.txt $output/warp_A.bin
python python_scripts/warp_pred.py $output/warp_B.bin $output/warp_A.bin $data/test_x.txt $data/mention.txt $output/warp_predictions $max_dep $delimter
echo "###########Baseline Performance"
python python_scripts/warp_eval.py $data/mention_type_test.txt $output/BBN/prediction_null_null_perceptron.txt
echo "###########Warp Performance"
python python_scripts/warp_eval.py $data/mention_type_test.txt $output/warp_predictions
python python_scripts/prediction_intext.py $raw_data $output/warp_predictions_intext $data $output/warp_predictions
python python_scripts/test_y2map.py $data $data/test_y.txt $output/gold.txt
python python_scripts/prediction_intext.py $raw_data $output/gold_intext $data $output/gold.txt
diff $output/warp_predictions_intext $output/gold_intext > $output/warp.comp
