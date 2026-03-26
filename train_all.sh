# configs=(Segformer_SegSTRONGC SETR_MLA_SegSTRONGC UNet_SegSTRONGC UNet_SegSTRONGC_AutoAugment SETR_MLA_SegSTRONGC_AutoAugment DeepLabV3p_SegSTRONGC UNetPlusPlus_SegSTRONGC Mask2Former_SegSTRONGC UNetPlusPlus_SegSTRONGC_AutoAugment Mask2Former_SegSTRONGC_AutoAugment)
# seed=(42 1234 2022 7 999)
configs=(Mask2Former_SegSTRONGC_Custom_Augment UNetPlusPlus_SegSTRONGC_Custom_Augment)
seed=(42 1234 2022 7 999)
test_domain=(bg_change regular blood smoke low_brightness)
for s in ${seed[@]}; do
  for c in ${configs[@]}; do
    echo Training $c with seed $s
    MODEL_PATH=../checkpoints/$c$s/
    # check if the model has been trained if not, then train
    if [ -f $MODEL_PATH"model_19.pth" ]; then
        echo "Model already trained, skipping training."
    else
        python train.py --config $c --seed $s
    fi
    for d in ${test_domain[@]}; do
        # check if the prediction file exists, if not then validate
        if [ ! -f "$MODEL_PATH$d/pred.npy" ]; then
          echo $MODEL_PATH$d/
          python validate.py --config $c --test True --model_path $MODEL_PATH"model_19.pth"  --domain $d --save_dir $MODEL_PATH$d/
        fi
    done
  done
done