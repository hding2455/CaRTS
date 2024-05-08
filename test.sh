#NET=Segformer_SegSTRONGC
#CHECKPOINT=checkpoints/segformer_segstrongc/model_39.pth
echo $NET BG Change
python validate.py --config $NET --model_path $CHECKPOINT --test True --domain bg_change --save_dir /workspace/data/SegSTRONG-C/results/bg_change/$NET
echo $NET Regular
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain regular --save_dir /workspace/data/SegSTRONG-C/results/regular/$NET
echo $NET Blood
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain blood --save_dir /workspace/data/SegSTRONG-C/results/bleeding/$NET
echo $NET Smoke
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain smoke --save_dir /workspace/data/SegSTRONG-C/results/smoke/$NET
echo $NET Low Brightness
python validate.py --config $NET --model_path $CHECKPOINT  --test True --domain low_brightness --save_dir /workspace/data/SegSTRONG-C/results/low_brightness/$NET
