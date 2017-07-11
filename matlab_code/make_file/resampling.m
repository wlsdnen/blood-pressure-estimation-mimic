function [ output_args ] = resampling( signal, len )
%RESAMPLING �� �Լ��� ��� ���� ��ġ
%   �ڼ��� ���� ��ġ

if length(signal) < len
    
    signal(end+1 : len) = 0;

end

output_args = signal;

end

