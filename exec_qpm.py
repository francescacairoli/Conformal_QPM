import copy
import time
import argparse
import pandas as pd
from QR import * # NN architecture to learn quantiles
from CQR import *
from CB_CQR import * # CC_CQR older version
from Loc_CQR import *
from Loc_CB_CQR import *
from utils import * # import-export methods
from Dataset import *
from TrainQR_multiquantile import *

# for the sake of reproducibility we fix the seeds
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--model_dim", default=2, type=int, help="Dimension of the model")
parser.add_argument("--model_prefix", default="MRH", type=str, help="Prefix of the model name")
parser.add_argument("--n_epochs", default=500, type=int, help="Nb of training epochs for QR")
parser.add_argument("--n_hidden", default=20, type=int, help="Nb of hidden nodes per layer")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
parser.add_argument("--qr_training_flag", default=True, type=eval, help="training flag")
parser.add_argument("--xavier_flag", default=False, type=eval, help="Xavier random weights initialization")
parser.add_argument("--scheduler_flag", default=False, type=eval, help="scheduler flag")
parser.add_argument("--opt", default="Adam", type=str, help="Optimizer")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Drop-out rate")
parser.add_argument("--alpha", default=0.1, type=float, help="quantiles significance level")
parser.add_argument("--property_idx", default=0, type=int, help="Identifier of the property to monitor (-1 denotes that the property is wrt all variables)")
parser.add_argument("--type_localizer", default="gauss", type=str, help="Type of localizer: gauss or knn")
parser.add_argument("--eps", default=0.1, type=float)
parser.add_argument("--knn", default=100, type=int)
args = parser.parse_args()
print(args)

print(torch.cuda.device_count())

model_name = args.model_prefix+str(args.model_dim)

print("Model name = ", model_name, " Model dim = ", args.model_dim)

trainset_fn, calibrset_fn, testset_fn, ds_details = import_filenames_w_dim(model_name, args.model_dim)
n_train_states, n_cal_states, n_test_states, cal_hist_size, test_hist_size = ds_details


quantiles = np.array([args.alpha/2, 0.5,  1-args.alpha/2]) # LB, MEDIAN, UB
nb_quantiles = len(quantiles)

idx_str = f'QPM_#{args.property_idx}_Dropout{args.dropout_rate}_multiout_opt=_{args.n_hidden}hidden_{args.n_epochs}epochs_{nb_quantiles}quantiles_3layers_alpha{args.alpha}_lr{args.lr}'


print(f"Models folder = Models/{model_name}/ID_{idx_str}")
print(f"Results folder = Results/{model_name}/ID_{idx_str}")

print(f"	Quantiles = {quantiles}")
print(f"	Property idx = {args.property_idx}")


print(f"Training settings: n_epochs = {args.n_epochs}, lr = {args.lr}, batch_size = {args.batch_size}")

# import data
dataset = Dataset(property_idx=args.property_idx, comb_flag=False, trainset_fn=trainset_fn, testset_fn=testset_fn, 
			calibrset_fn=calibrset_fn, alpha=args.alpha, n_train_states=n_train_states, n_cal_states=n_cal_states, 
			n_test_states=n_test_states, hist_size=cal_hist_size, test_hist_size=test_hist_size)
eqr_width = dataset.load_data()

print(f"Test EQR width = {eqr_width}")

# Train the QR
qr = TrainQR(model_name, dataset, idx = idx_str, cal_hist_size  = cal_hist_size, test_hist_size = test_hist_size, quantiles = quantiles, opt = args.opt, n_hidden = args.n_hidden, xavier_flag = args.xavier_flag, scheduler_flag = args.scheduler_flag, drop_out_rate = args.dropout_rate)
qr.initialize()

if args.qr_training_flag:
	start_time = time.time()
	qr.train(args.n_epochs, args.batch_size, args.lr)
	end_time = time.time()-start_time
	qr.save_model()
	print(f'Training time for {model_name}-#{args.property_idx} with {args.n_epochs} epochs = {end_time}')
else:
	qr.load_model(args.n_epochs)

qr.qr_model.eval()

# Obtain CQR intervals given the trained QR
cqr = CQR(dataset.X_cal, dataset.R_cal, qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
cpi_test, pi_test = cqr.get_cpi(dataset.X_test, pi_flag = True)

cc_cqr = CC_CQR(dataset.X_cal, dataset.R_cal, qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
_, cc_cpi_test = cc_cqr.get_cpi(dataset.X_test)
_, pos_cpi_test, neg_cpi_test = cc_cqr.get_cc_cpi(dataset.X_test)

loc_cqr = Loc_CQR(Xc=dataset.X_cal, Yc=dataset.R_cal, type_local = args.type_localizer, knn = args.knn, eps=args.eps, trained_qr_model=qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
loc_cqr.get_calibr_scores()

_, loc_cpi_test = loc_cqr.get_cpi(dataset.X_test)

loc_cb_cqr = Loc_CB_CQR(Xc=dataset.X_cal, Yc=dataset.R_cal, type_local = args.type_localizer, knn = args.knn, eps=args.eps, trained_qr_model=qr.qr_model, test_hist_size = test_hist_size, cal_hist_size = cal_hist_size)
loc_cb_cqr.get_calibr_scores()

_, loc_cb_cpi_test = loc_cb_cqr.get_cpi(dataset.X_test)

_ = cqr.coverage_distribution(dataset.R_test, cpi_test, qr.results_path, 1-args.alpha)
_ = cqr.cc_coverage_distribution(dataset.R_test, cpi_test, qr.results_path, 1-args.alpha)

_ = cc_cqr.coverage_distribution(dataset.R_test,cc_cpi_test, qr.results_path, 1-args.alpha)
_ = cc_cqr.cc_coverage_distribution(dataset.R_test, cc_cpi_test, qr.results_path, 1-args.alpha)

_ = loc_cqr.coverage_distribution(dataset.R_test, loc_cpi_test, qr.results_path, 1-args.alpha)
_ = loc_cqr.cc_coverage_distribution(dataset.R_test, loc_cpi_test, qr.results_path, 1-args.alpha)

_ = loc_cb_cqr.coverage_distribution(dataset.R_test, loc_cb_cpi_test, qr.results_path, 1-args.alpha)
_ = loc_cb_cqr.cc_coverage_distribution(dataset.R_test, loc_cb_cpi_test, qr.results_path, 1-args.alpha)

pi_coverage, pi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, pi_test)
print("pi_coverage = ", pi_coverage, "pi_efficiency = ", pi_efficiency)
pi_correct, pi_uncertain, pi_wrong, pi_fp = cqr.compute_accuracy_and_uncertainty(pi_test, dataset.L_test)
print("pi_correct = ", pi_correct, "pi_uncertain = ", pi_uncertain, "pi_wrong = ", pi_wrong, "pi_fp = ", pi_fp)

cpi_coverage, cpi_efficiency = cqr.get_coverage_efficiency(dataset.R_test, cpi_test)
print("cpi_coverage = ", cpi_coverage, "cpi_efficiency = ", cpi_efficiency)
cpi_correct, cpi_uncertain, cpi_wrong, cpi_fp = cqr.compute_accuracy_and_uncertainty(cpi_test, dataset.L_test)
print("cpi_correct = ", cpi_correct, "cpi_uncertain = ", cpi_uncertain, "cpi_wrong = ", cpi_wrong, "cpi_fp = ", cpi_fp)

#cc_cpi_coverage, cc_cpi_efficiency = cc_cqr.get_coverage_efficiency_coupled(dataset.R_test, pos_cpi_test, neg_cpi_test)
cc_cpi_coverage, cc_cpi_efficiency = cc_cqr.get_coverage_efficiency(dataset.R_test, cc_cpi_test)
print("cc_cpi_coverage = ", cc_cpi_coverage, "cc_cpi_efficiency = ", cc_cpi_efficiency)
cc_cpi_correct, cc_cpi_uncertain, cc_cpi_wrong, cc_cpi_fp = cc_cqr.compute_accuracy_and_uncertainty(cc_cpi_test, dataset.L_test)
print("cc_cpi_correct = ", cc_cpi_correct, "cc_cpi_uncertain = ", cc_cpi_uncertain, "cc_cpi_wrong = ", cc_cpi_wrong, "cc_cpi_fp = ", cc_cpi_fp)

loc_cpi_coverage, loc_cpi_efficiency = loc_cqr.get_coverage_efficiency(dataset.R_test, loc_cpi_test)
print("loc_cpi_coverage = ", loc_cpi_coverage, "loc_cpi_efficiency = ", loc_cpi_efficiency)
loc_cpi_correct, loc_cpi_uncertain, loc_cpi_wrong, loc_cpi_fp = loc_cqr.compute_accuracy_and_uncertainty(loc_cpi_test, dataset.L_test)
print("loc_cpi_correct = ", loc_cpi_correct, "loc_cpi_uncertain = ", loc_cpi_uncertain, "loc_cpi_wrong = ", loc_cpi_wrong, "loc_cpi_fp = ", loc_cpi_fp)

cqr.plot_errorbars(dataset.R_test, pi_test, cpi_test, "predictive intervals", qr.results_path, 'pred_interval')

cc_cqr.plot_cc_errorbars(dataset.R_test, pi_test, cpi_test, pos_cpi_test, neg_cpi_test, "predictive intervals", qr.results_path, 'pred_interval')
cc_cqr.plot_errorbars(dataset.R_test, pi_test, cpi_test, cc_cpi_test, "predictive intervals", qr.results_path, 'pred_interval')

loc_cqr.plot_loc_errorbars(dataset.R_test, pi_test, cpi_test, loc_cpi_test, "predictive intervals", qr.results_path, 'pred_interval')

d = {model_name:['QR', 'CQR', 'CC_CQR', 'Loc_CQR'],'correct': [pi_correct, cpi_correct, cc_cpi_correct, loc_cpi_correct],
	'uncertain': [pi_uncertain, cpi_uncertain,  cc_cpi_uncertain, loc_cpi_uncertain],
	'wrong':[pi_wrong, cpi_wrong,  cc_cpi_wrong, loc_cpi_wrong], 'FP':[pi_fp, cpi_fp,  cc_cpi_fp, loc_cpi_fp],
	'coverage':[pi_coverage, cpi_coverage, cc_cpi_coverage, loc_cpi_coverage],
	'efficiency': [pi_efficiency, cpi_efficiency, cc_cpi_efficiency, loc_cpi_efficiency],
	'EQR width': [eqr_width, '-', '-', '-']}
df = pd.DataFrame(data=d)
print('Table of results:\n ',df)
out_tables_path = f"out/tables/{args.model_prefix}"
os.makedirs(out_tables_path, exist_ok=True)
df.to_csv(out_tables_path+f"/{model_name}_#{args.property_idx}_results.csv", index=False)

results_list = ["Id = ", idx_str, "\n", "\n epsilon=", str(args.eps), "\n Quantiles = ", str(quantiles), "\n tau = ", str(cqr.tau), "\n",
"\n pi_correct = ", str(pi_correct), "\n pi_uncertain = ", str(pi_uncertain), "\n pi_wrong = ", str(pi_wrong),"\n pi_fp = ", str(pi_fp),"\n pi_coverage = ", str(pi_coverage), "\n pi_efficiency = ", str(pi_efficiency),
"\n",
"\n eqr_width = ", str(eqr_width),
"\n",
"\n cpi_correct = ", str(cpi_correct), "\n cpi_uncertain = ", str(cpi_uncertain), "\n cpi_wrong = ", str(cpi_wrong),"\n cpi_fp = ", str(cpi_fp),"\n cpi_coverage = ", str(cpi_coverage), "\n cpi_efficiency = ", str(cpi_efficiency),
"\n",
#"\n cc_cpi_correct = ", str(cc_cpi_correct), "\n cc_cpi_uncertain = ", str(cc_cpi_uncertain), "\n cc_cpi_wrong = ", str(cc_cpi_wrong),"\n cc_cpi_fp = ", str(cc_cpi_fp),
"\n cc_cpi_coverage = ", str(cc_cpi_coverage), "\n cc_cpi_efficiency = ", str(cc_cpi_efficiency),
"\n",
"\n loc_cpi_correct = ", str(loc_cpi_correct), "\n loc_cpi_uncertain = ", str(loc_cpi_uncertain), "\n loc_cpi_wrong = ", str(loc_cpi_wrong),"\n loc_cpi_fp = ", str(loc_cpi_fp),"\n loc_cpi_coverage = ", str(loc_cpi_coverage), "\n loc_cpi_efficiency = ", str(loc_cpi_efficiency)
]

save_results_to_file(results_list, qr.results_path, extra_info=f'_eps={args.eps}')
print(qr.results_path)