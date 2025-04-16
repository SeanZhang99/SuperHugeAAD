function result = find_trialwise_acc(data_table,dataset_id,stage)
% 从表格中提取所有 trial_xxx_acc 格式的列，计算每列最大值
% 输入:
%   data_table - cell 数组（使用 readcell() 从 CSV 读取）
% 输出:
%   result - cell 数组，包含每个匹配列的名称和最大值

    header = data_table(1, :);
    data = data_table(2:end, :);

    result = {};  % 初始化结果

    pattern = sprintf("detail/%s/dataset-%03d-subject-(\\d+)-trial-\\d+_acc",stage,dataset_id);

    for col = 1:numel(header)
        col_name = header{col};
        if ischar(col_name) || isstring(col_name)
            tokens = regexp(col_name, pattern, 'tokens', 'once');
            if ~isempty(tokens)
                subject = str2double(tokens{1});
                % 提取该列数据
                col_data = data(:, col);
                % 转成数值
                numeric_data = cellfun(@double,col_data);
                numeric_data = numeric_data(~isnan(numeric_data));
                % 计算最大值
                if ~isempty(numeric_data)
                % 加入结果
                    result(end+1, :) = {col_name;subject;numeric_data};
                end
            end
        end
    end
end
