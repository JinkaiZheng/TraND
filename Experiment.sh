# CASIA-B -> OULP
python main.py --gpus=0,1,2,3 --load-pretrained=True \
--max-epoch=200 --start-round=1 --max-round=4 \
--loss-init-rate=0.01 --ANs-size=1 \
--source="casia-b" --target="oulp" \
--log-name='CASIA2OULP'

# OULP -> CASIA-B
#python main.py --gpus=0,1,2,3 --load-pretrained=True \
#--max-epoch=200 --start-round=1 --max-round=4 \
#--loss-init-rate=0.01 --ANs-size=1 \
#--source="oulp" --target="casia-b" \
#--log-name='OULP2CASIA'