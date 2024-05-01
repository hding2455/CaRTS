NET=Segformer_SegSTRONGC
CHECKPOINT=checkpoints/segformer_segstrongc/model_39.pth
echo $NET BG Change
python validate.py --config $NET --model_path $CHECKPOINT --test True --domain bg_change --save_dir results/bg_change/$NET
echo $NET Regular
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain regular --save_dir results/regular/$NET
echo $NET Blood
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain blood --save_dir results/bleeding/$NET
echo $NET Smoke
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain smoke --save_dir results/smoke/$NET
echo $NET Low Brightness
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain low_brightness --save_dir results/low_brightness/$NET
