% index = dlmread ('rmse_6.8276_i.txt');
% prediction = dlmread ('rmse_6.8276_p.txt');
% target = dlmread ('rmse_6.8276_y.txt');
% 
% error = target - prediction;
% error = [error, index, target, prediction];
% 
% for i = 5:5:50
% %     Systolic
%     eval(sprintf ('error%02d = error(abs(error(:, 1)) > %d, :);', i, i))
% %     Diastolic
%     eval(sprintf ('error%02d = error%02d(abs(error%02d(:, 2)) > %d, :);', i, i, i, i))
%     
%     eval(sprintf ('error%02d = sortrows (error%02d, 3);', i, i));
% 
% end
index       = dlmread ('rmse_6.6859_i.txt');
prediction  = dlmread ('rmse_6.6859_p.txt');
target      = dlmread ('rmse_6.6859_y.txt');

error = target - prediction;
error = [error, index, target, prediction];

range = 5;

for i = 0:range:100
    
%     Systolic
    eval(sprintf ('error%02ds   = error( abs( error(:, 1) ) > %d & abs(error(:, 1)) < %d, :);', i, i-1, i+range))
    
%     Diastolic
    eval(sprintf ('error%02dd   = error( abs( error(:, 2) ) > %d & abs(error(:, 2)) < %d, :);', i, i-1, i+range))
    
%     Systolic & Diastolic
    eval(sprintf ('error%02ds_d = error%02ds(abs(error%02ds(:, 2)) > %d & abs(error%02ds(:, 2)) < %d, :);', i, i, i, i-1, i, i+range))
    
%     Sort by cell index
    eval(sprintf ('error%02ds   = sortrows (error%02ds, 3);', i, i));
    eval(sprintf ('error%02dd   = sortrows (error%02dd, 3);', i, i));
    eval(sprintf ('error%02ds_d = sortrows (error%02ds_d, 3);', i, i));

end

HistErrors(error(:, 1), error(:, 2))
CdfErrors(error(:, 1), error(:, 2))