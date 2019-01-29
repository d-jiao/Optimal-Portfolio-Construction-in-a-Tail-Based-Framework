res = csvread('.\\data\\res.csv', 1, 1);
upper_quantile = [0.9721086 0.9750837 0.9750837 0.9888434];
lower_quantile = [0.01487542 0.03756043 0.01375976 0.01190033];

upper_threshold = [0.03, -0.15, -0.02, 0.01];
lower_threshold = [-0.04, -0.19, -0.08, -0.043];

shsz = res(:, 1);
fitted = paretotails(shsz, lower_quantile(1), upper_quantile(1));
tdist = fitdist(shsz, 'tLocationScale');
ndist = fitdist(shsz, 'Normal');
[edist, ex] = ecdf(shsz);

x_r = linspace(upper_threshold(1), max(shsz));
scatter(ex(min(find(ex > upper_threshold(1)) : end), edist(min(find(ex > upper_threshold(1))) : end));
hold on
plot(x_r, cdf(tdist, x_r));
plot(x_r, cdf(ndist, x_r));
plot(x_r, cdf(fitted, x_r));
legend('Empirical', 'Student''s t', 'Normal', 'Pareto', 'Location', 'southeast');

x_l = linspace(min(shsz), lower_threshold(1));
scatter(ex(1 : max(find(ex <= lower_threshold(1)))), edist(1 : max(find(ex <= lower_threshold(1)))));
hold on
plot(x_l, cdf(tdist, x_l));
plot(x_l, cdf(ndist, x_l));
plot(x_l, cdf(fitted, x_l));
legend('Empirical', 'Student''s t', 'Normal', 'Pareto', 'Location', 'southeast');
