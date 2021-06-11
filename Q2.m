% Generates N samples from a specified Gaussian Mixture PDF
% then uses EM algorithm to estimate the parameters along
% with BIC to select the model order, which is the number
% of Gaussian components for the model.
function [data_trueMs, data_truegmm, data_bestMs, data_bestGMMs, data_dataset] =
HW_2_fitgmdist(method, jobidx)
dim = 2;
maxExp = 10;
%generate data samples
dataset_count = 4;
% method = 0;
% Evaluate BIC for candidate model orders
tic
for datasetIdx = 1:dataset_count
N = 10^(1+datasetIdx);
maxM = floor(N^(1/2)); % arbitrarily selecting the maximum model using this rule
if(maxM > 15)
maxM = 15;
end
bestMs = zeros(1, maxExp);
bestGMMs = {};
dataset_name = num2str(N)
trueMs = zeros(1,maxExp);
dataset = zeros(N,dim,maxExp);
parfor experiment = 1:maxExp
display(experiment); %#ok<*NOPTS>
jobidx
trueMs(experiment) = 10;
rng shuffle;
[dataset(:,:,experiment), truegmm{experiment}] =
generateRandGMMData(trueMs(experiment), dim, N);
if method(1) == 1
[bestMs(experiment), bestGMMs{experiment}] =
runBICexperiment(dataset(:,:,experiment), N, dim, maxM);
elseif method(1) == 0
[bestMs(experiment), mu, Sigma, alpha] =
runkfoldexperiment_fitgmdist(dataset(:,:,experiment), 10, N, dim, maxM);
bestGMMs{experiment} = gmdistribution(mu', Sigma, alpha);
end
end
data_trueMs.(['N', dataset_name]) = trueMs;
data_truegmm.(['N', dataset_name]) = truegmm;
data_bestMs.(['N', dataset_name]) = bestMs;
data_bestGMMs.(['N', dataset_name]) = bestGMMs;
data_dataset.(['N', dataset_name]) = dataset;
end
toc
if method(1) == 1
save(['BIC_', num2str(jobidx)]);
elseif method(1) == 0
save(['kfold_', num2str(jobidx)]);
end
end
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
function [bestM, bestGMM] = runBICexperiment(dataset, N, dim, maxM)
nSamples = dim*N;
nParams = zeros(1,maxM);
BIC = zeros(1,maxM);
neg2logLikelihood = zeros(1,maxM);
for M = 1:maxM
nParams(1,M) = (M-1) + dim*M + M*(dim+nchoosek(dim,2));
options = statset('MaxIter',10000);
gm{M} = fitgmdist(dataset,M,'Replicates',1, 'RegularizationValue',1e-10,'Options',options);
% [mu, Sigma, alpha] = fitgmm(dim, N, M, dataset);
% gm{M} = gmdistribution(mu', Sigma, alpha);
neg2logLikelihood(1,M) = -2*sum(log(pdf(gm{M},dataset)));
BIC(1,M) = neg2logLikelihood(1,M) + nParams(1,M)*log(nSamples);
end
[~,bestM] = min(BIC);
bestGMM = gm{bestM};
end
function [mu, Sigma, alpha] = fitgmm(dim, N, M, dataset)
delta = 1e-2; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
% Initialize the GMM to randomly selected samples
x = dataset';
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest
mean
Sigma = zeros(dim, dim, M);
temp = zeros(M, N);
SigmaNew = zeros(dim, dim, M);
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
%alpha(1,m) = find(assignedCentroidLabels==m)/N;
Sigma(:,:,m) = cov(x(:,assignedCentroidLabels==m)') + regWeight*eye(dim,dim);
end
total_delta = 1e8;
while total_delta > delta
for l = 1:M
temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
end
temp_sum = repmat(sum(temp, 1),M,1);
plgivenx = temp./temp_sum;
alphaNew = mean(plgivenx,2)';
w = plgivenx./repmat(sum(plgivenx,2),1,N);
muNew = x*w';
for l = 1:M
v = x-repmat(muNew(:,l),1,N);
u = repmat(w(l,:),dim,1).*v;
SigmaNew(:,:,l) = u*v' + regWeight*eye(dim,dim); % adding a small regularization term
end
Dalpha = sum(abs(alphaNew-alpha));
Dmu = sum(sum(abs(muNew-mu)));
DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
total_delta = Dalpha+Dmu+DSigma(1,1,1);
alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
end
end
function A = generateSPDmatrix(n)
% A = rand(n,n); % generate a random n x n matrix
% A = A*A';
% A = A + n*eye(n);
A = (rand(1,1)*200).*eye(n);
end
function [data, gmdist] = generateRandGMMData(Order, dim, N)
mu = [linspace(-200,200,Order); zeros(1,Order)];
sigma = zeros(dim,dim,Order);
for m = 1:Order
sigma(:,:,m) = (rand(1,1)*100).*eye(dim);
end
gmdist = gmdistribution(mu',sigma,ones(1,Order)/Order);
data = random(gmdist,N);
end
function [bestM, mu, Sigma, alpha] = runkfoldexperiment_fitgmdist(dataset, K, N, dim, maxM)
partitions_idx_start = ceil(linspace(0,N,K+1));
indPartitionLimits = zeros(K, 2);
for k = 1:K
indPartitionLimits(k,:) = [partitions_idx_start(k)+1,partitions_idx_start(k+1)];
end
loglikelyood_M = zeros(1,maxM);
options = statset('MaxIter',10000);
for M = 1:maxM
loglikelyhood = zeros(1,K);
for k= 1:K
indValidate = indPartitionLimits(k,1):indPartitionLimits(k,2);
x = dataset';
xValidate = x(:, indValidate); % Using folk k as validation set
if k == 1
indTrain = indPartitionLimits(k+1,1):N;
elseif k == K
indTrain = 1:indPartitionLimits(k-1,2);
else
indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):N];
end
xTrain = x(:, indTrain); % using all other folds as training set
Ntrain = length(indTrain);
gm = fitgmdist(xTrain',M,'Replicates',10, 'RegularizationValue',1e-1,'Options',options);
% [mu, Sigma, alpha] = fitgmm(dim, Ntrain, M, xTrain');
loglikelyhood(k) = sum(log(pdf(gm, xValidate')));
end
loglikelyood_M(M) = mean(loglikelyhood);
end
[~, bestM] = max(loglikelyood_M);
finalgm = fitgmdist(dataset,bestM,'Replicates',10, 'RegularizationValue',1e-1,'Options',options);
mu = finalgm.mu';
Sigma = finalgm.Sigma;
alpha = finalgm.ComponentProportion;
% [finalgm.mu, finalgm.Sigma, finalgm.alpha] = fitgmm(dim, N, bestM, dataset);
end
function [mu, Sigma, alpha] = fitgmm(dim, N, M, dataset)
delta = 1e-2; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
% Initialize the GMM to randomly selected samples
x = dataset';
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest
mean
Sigma = zeros(dim, dim, M);
temp = zeros(M, N);
SigmaNew = zeros(dim, dim, M);
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
%alpha(1,m) = find(assignedCentroidLabels==m)/N;
Sigma(:,:,m) = cov(x(:,assignedCentroidLabels==m)') + regWeight*eye(dim,dim);
end
total_delta = 1e8;
while total_delta > delta
for l = 1:M
temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
end
temp_sum = repmat(sum(temp, 1),M,1);
plgivenx = temp./temp_sum;
alphaNew = mean(plgivenx,2)';
w = plgivenx./repmat(sum(plgivenx,2),1,N);
muNew = x*w';
for l = 1:M
v = x-repmat(muNew(:,l),1,N);
u = repmat(w(l,:),dim,1).*v;
SigmaNew(:,:,l) = u*v' + regWeight*eye(dim,dim); % adding a small regularization term
end
Dalpha = sum(abs(alphaNew-alpha));
Dmu = sum(sum(abs(muNew-mu)));
DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
total_delta = Dalpha+Dmu+DSigma(1,1,1);
alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
end
end
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end