function [ output_args ] = resampling( signal, len )
%RESAMPLING 이 함수의 요약 설명 위치
%   자세한 설명 위치

if length(signal) < len
    
    signal(end+1 : len) = 0;

end

output_args = signal;

end

