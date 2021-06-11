close all, clear all
% define dataset parameters
row = 3; col = 2;
scatter_plot_index_gaussian = 1;
roc_gaussian_plot_index = 2;
scatter_plot_index_logistic_linear = 3;
roc_logistic_linear_plot_index = 4;
scatter_plot_index_logistic_quadratic = 5;
roc_logistic_quadratic_plot_index = 6;
parameters.alpha{1} = [0.5, 0.5];
parameters.mu{1} = [5 0; 0 4]';
parameters.Sigma{1}(:,:,1) = [4 0;0 2];
parameters.Sigma{1}(:,:,2) = [1 0;0 3];
parameters.alpha{2} = 1;
parameters.Sigma{2}(:,:,1) = [2 0;0 2];
parameters.mu{2} = [3 2];
parameters.priors = [0.6 0.4];
% generate datasets
data.train_100 = generateGMMData(parameters, 100);
data.train_1k = generateGMMData (parameters, 1000);
data.train_10k = generateGMMData(parameters, 10000);
[data.valid_20k, gmmtrue] = generateGMMData(parameters, 20000);
fn = fieldnames(data);
% gaussian w true pdf
tau_true = log(parameters.priors(1)/parameters.priors(2));
a = likelyhood_ratio(gmmtrue, data.valid_20k.features);
results.gaussian.truepdf = ERMeval(descriminantScores.truepdf, data.valid_20k.labels,
tau_true);
results.gaussian.roc.truepdf = ROCcurve(descriminantScores.truepdf,data.valid_20k.labels);
figure(1)
subplot(row, col, scatter_plot_index_gaussian);
gmm_scatter_plot(data.valid_20k);
contour_plot_gaussian(data.valid_20k, 500, gmmtrue, tau_true);
title('Scatter plot for D_{valid}^{20k} with contour plot of optimum classifier');
% gaussian w training
for k=1:numel(fn)-1
display(['fitting gmm parameters from ', fn{k}]);
gmm.(fn{k}) = estimate_pdf_params(data.(fn{k}));
display(['evaluating descriminant scores ', fn{k}]);
descriminantScores.gaussian.(fn{k}) = likelyhood_ratio(gmm.(fn{k}), data.valid_20k.features);
results.gaussian.roc.(fn{k}) =
ROCcurve(descriminantScores.gaussian.(fn{k}),data.valid_20k.labels);
end
figure(1)
subplot(row, col, roc_gaussian_plot_index);
plot(results.gaussian.roc.truepdf.pfp, results.gaussian.roc.truepdf.ptp), hold on
plot(results.gaussian.roc.truepdf.min_pfp, results.gaussian.roc.truepdf.min_ptp, '*r'), hold on
plot_roc(fn, results.gaussian.roc);
legend('guassian pdf model w true params', '\gamma_{opt}', ...
'gaussian 100', 'gaussian 1k', 'gaussian 10k');
title('Roc Plot for gaussian classifiers');
% logistic
% linear training
for k=1:numel(fn)-1
display(['fitting logistic-linear-function-based model parameters from ', fn{k}]);
z = linear_feature_representation_for_logistic_fn(data.(fn{k}).features);
[theta.linear.(fn{k}),cost] = fit_logistic_fn(data.(fn{k}), z);
display(['evaluating descriminant scores ', fn{k}]);
z = linear_feature_representation_for_logistic_fn(data.valid_20k.features);
descriminantScores.logistic.linear.(fn{k}) = likelyhood_ratio_logistic(theta.linear.(fn{k}), z);
results.logistic.linear.roc.(fn{k}) =
ROCcurve(descriminantScores.logistic.linear.(fn{k}),data.valid_20k.labels);
end
subplot(row, col, scatter_plot_index_logistic_linear);
gmm_scatter_plot(data.valid_20k);
contour_plot_logistic(data.valid_20k, @linear_feature_representation_for_logistic_fn, 500,
theta.linear.train_10k);
title('Scatter plot for D_{valid}^{20k} with contour plot of log-linear classifier');
subplot(row, col, roc_logistic_linear_plot_index);
plot(results.gaussian.roc.truepdf.pfp, results.gaussian.roc.truepdf.ptp, 'LineWidth',2), hold on
plot_roc(fn, results.logistic.linear.roc);
legend('guassian pdf model w true params', ...
'logistic linear 100', 'logistic linear 1k', 'logistic linear 10k');
title('Roc Plot for log-linear classifiers');
% quadritic training
for k=1:numel(fn)-1
display(['fitting logistic-quadratic-function-based model parameters from ', fn{k}]);
z = quadratic_feature_representation_for_logistic_fn(data.(fn{k}).features);
[theta.linear.(fn{k}),cost] = fit_logistic_fn(data.(fn{k}), z);
display(['evaluating descriminant scores ', fn{k}]);
z = quadratic_feature_representation_for_logistic_fn(data.valid_20k.features);
descriminantScores.logistic.(fn{k}) = likelyhood_ratio_logistic(theta.linear.(fn{k}), z);
results.logistic.quadratic.roc.(fn{k}) =
ROCcurve(descriminantScores.logistic.(fn{k}),data.valid_20k.labels);
end
subplot(row, col, scatter_plot_index_logistic_quadratic);
gmm_scatter_plot(data.valid_20k);
contour_plot_logistic(data.valid_20k, @quadratic_feature_representation_for_logistic_fn, 500,
theta.linear.train_10k);
title('Scatter plot for D_{valid}^{20k} with contour plot of log-quadratic classifier');
subplot(row, col, roc_logistic_quadratic_plot_index);
plot(results.gaussian.roc.truepdf.pfp, results.gaussian.roc.truepdf.ptp, 'LineWidth',2), hold on
plot_roc(fn, results.logistic.quadratic.roc);
lgnd = legend('guassian pdf model w true params', ...
'logistic quadratic 100', 'logistic quadratic 1k', 'logistic quadratic 10k');
title('Roc Plot for log-quadratic classifiers');
% FUNCTIONS
function contour_plot_gaussian(grid_data, gridSize, gmmtrue, tau_true)
x1Grid = linspace(floor(min(grid_data.features(1,:))),ceil(max(grid_data.features(1,:))),gridSize);
x2Grid = linspace(floor(min(grid_data.features(2,:))),ceil(max(grid_data.features(2,:))),gridSize);
[a, b] = meshgrid(x1Grid,x2Grid);
meshfeatures = [a(:)';b(:)'];
discriminantScoreGridValues = log(pdf(gmmtrue{2}, meshfeatures')) - log(pdf(gmmtrue{1},
meshfeatures')) - log(tau_true);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,gridSize,gridSize);
contour(x1Grid,x2Grid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]
); % plot equilevel contours of the discriminant function
lgnd = legend('Class 0','Class 1', 'Contours of discriminant function');
lgnd.Location = 'southeast';
end
function contour_plot_logistic(grid_data, x_rep, gridSize, theta)
x1Grid = linspace(floor(min(grid_data.features(1,:))),ceil(max(grid_data.features(1,:))),gridSize);
x2Grid = linspace(floor(min(grid_data.features(2,:))),ceil(max(grid_data.features(2,:))),gridSize);
[a, b] = meshgrid(x1Grid,x2Grid);
meshfeatures = [a(:)';b(:)'];
z = x_rep(meshfeatures);
discriminantScoreGridValues = log(logistic_model(theta,z)) - log((1-logistic_model(theta,z)));
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,gridSize,gridSize);
contour(x1Grid,x2Grid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]
); % plot equilevel contours of the discriminant function
lgnd = legend('Class 0','Class 1', 'Contours of discriminant function');
lgnd.Location = 'southeast';
end
function lgnd = plot_roc(fieldnames, results)
fn = fieldnames;
for k=1:numel(fn)-1
plot(results.(fn{k}).pfp,results.(fn{k}).ptp), hold on
end
xlabel('P(False+)'),ylabel('P(True+)'), title('ROC Curves'),
lgnd = legend;
end
function z = linear_feature_representation_for_logistic_fn(features)
N = size(features, 2);
z=[ones(N,1) features'];
end
function z = quadratic_feature_representation_for_logistic_fn(features)
N = size(features, 2);
z=[ones(N,1) features(1,:)' features(2,:)' (features(1,:).^2)' (features(1,:).*features(2,:))'
(features(2,:).^2)'];
end
function [theta,cost] = fit_logistic_fn(data, z)
options = optimset('MaxIter',10000, 'MaxFunEvals', 10000);
N = size(data.features, 2);
theta_init = zeros(1, size(z,2));
[theta,cost]=fminsearch(@(t)(costfunc(t, z, data.labels, N)),theta_init,options);
end
function gmm = estimate_pdf_params(data)
options = statset('MaxIter',10000);
gmm{1} = fitgmdist(data.features(:, find(data.labels==0))',2,'Replicates',30,'Options',options);
mu = mean(data.features(:, find(data.labels==1)),2)';
sigma = cov(data.features(:, find(data.labels==1))');
alpha = 1;
gmm{2} = gmdistribution(mu,sigma,alpha);
end
function descriminantScores = likelyhood_ratio(gmm, features)
descriminantScores = log(pdf(gmm{2}, features')./pdf(gmm{1}, features'))';
end
function descriminantScores = likelyhood_ratio_logistic(theta, features)
descriminantScores = log(logistic_model(theta,features)./(1-logistic_model(theta,features)))';
end
function gmm_scatter_plot(data)
scatter(data.features(1, data.labels==0), data.features(2, data.labels==0), 'b'), hold on
scatter(data.features(1, data.labels==1), data.features(2, data.labels==1), 'r'), hold on
lgnd = legend('Class 0','Class 1');
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'),
end
function [data, gmtrue] = generateGMMData(parameters, N)
data.labels = (rand(1,N)>=parameters.priors(1));
dim = size(parameters.mu{1}, 1);
data.features = zeros(dim,N);
for l = 1:size(parameters.priors,2)
gmtrue{l} = gmdistribution(parameters.mu{l},parameters.Sigma{l},parameters.alpha{l});
ind{l} = find(data.labels==l-1);
data.features(:,ind{l}) = random(gmtrue{l},size(ind{l},2))';
end
end
function [results] = ROCcurve(discriminantScores,labels)
[sortedScores,~] = sort(discriminantScores,'ascend');
thresholdList = [min(sortedScores)-eps,(sortedScores(1:end-1)+sortedScores(2:end))/2,
max(sortedScores)+eps];
ptp = zeros(1,length(thresholdList));
pfp = zeros(1,length(thresholdList));
perror = zeros(1,length(thresholdList));
parfor i = 1:length(thresholdList)
tau = thresholdList(i);
result = ERMeval(discriminantScores, labels, tau);
ptp(i) = result.ptp;
pfp(i) = result.pfp;
perror(i) = result.perror;
end
results.ptp = ptp;
results.pfp = pfp;
results.perror = perror;
[results.min_perror, min_ind] = min(perror);
results.min_threshold = thresholdList(min_ind(1));
results.min_ptp = ptp(min_ind(1));
results.min_pfp = pfp(min_ind(1));
end
function results = ERMeval(discriminantScores, labels, tau)
decisions = (discriminantScores >= tau);
results.ptp = length(find(decisions==1 & labels==1))/length(find(labels==1));
results.pfp = length(find(decisions==1 & labels==0))/length(find(labels==0));
results.perror = sum(decisions~=labels)/length(labels);
end
function cost=costfunc(theta,x,label,N)
h = logistic_model(theta,x);
cost=(-1/N)*(label*log(h)+(1-label)*log(1-h));
end
function h = logistic_model(theta,x)
h=1./(1+exp(-x*theta'));
end
