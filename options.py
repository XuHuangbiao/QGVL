import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--video-path', type=str, default='../action_assessment/rg_feat/swintx_avg_fps25_clip32')
parser.add_argument('--clip-num', type=int, default=95)

parser.add_argument('--train-label-path', type=str, default='../action_assessment/rg_feat/train.txt')
parser.add_argument('--test-label-path', type=str, default='../action_assessment/rg_feat/test.txt')

parser.add_argument('--action-type', type=str, default='Ball')
parser.add_argument('--score-type', type=str, default='Total_Score')

parser.add_argument('--model-name', type=str, default='action_net', help='name used to save model and logs')
parser.add_argument("--ckpt", default=None, help="ckpt for pretrained model")
parser.add_argument("--test", action='store_true', help="only evaluate, don't train")

parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)

parser.add_argument('--optim', type=str, default='sgd')

parser.add_argument("--lr-decay", type=str, default=None, help='use what decay scheduler')
parser.add_argument("--decay-rate", type=float, default=0.1, help="lr decay rate")
parser.add_argument("--warmup", type=int, default=10, help="warmup epoch")

parser.add_argument('--in_dim', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--n_encoder', type=int, default=1)
parser.add_argument('--n_decoder', type=int, default=1)
parser.add_argument('--n_query', type=int, default=4)
parser.add_argument('--score_type', type=str, default='TES')
parser.add_argument('--score_range', type=int, default=45)

parser.add_argument('--alpha_tl', type=float, default=1.0)
parser.add_argument('--margin', type=float, default=1.0)
parser.add_argument('--alpha_ce', type=float, default=0.5)
parser.add_argument('--alpha_mse', type=float, default=10.0)

parser.add_argument('--dropout', type=float, default=0.0)

# CLIP
parser.add_argument('--ARCH', type=str, default='ViT-B/16')
parser.add_argument('--DROP_PATH_RATE', type=float, default=0.)
parser.add_argument("--RETRAINED", type=bool, default=None)
parser.add_argument("--FIX_TEXT", type=bool, default=True)
parser.add_argument("--MULTI_VIEW_INFERENCE", type=bool, default=True)
parser.add_argument("--ZS_EVAL", type=bool, default=False)
parser.add_argument('--USE', type=str, default='both')
parser.add_argument("--PROMPT_MODEL", type=bool, default=False)
parser.add_argument('--N_CTX_VISION', type=int, default=0)
parser.add_argument('--N_CTX_TEXT', type=int, default=0)
parser.add_argument('--CTX_INIT', type=str, default='a video of a figure skating with')
# parser.add_argument('--CTX_INIT', type=str, default='a video of a rhythmic gymnastics with')
parser.add_argument('--PROMPT_DEPTH_VISION', type=int, default=0)
parser.add_argument('--PROMPT_DEPTH_TEXT', type=int, default=1)
parser.add_argument('--pretrained_clip_weight', type=str,
                    default='/data/xhb/CLIP-AQA/1/models/k400_clip_complete_finetuned_30_epochs.pth')
parser.add_argument('--action_list', type=str, default='./action-label.csv')
