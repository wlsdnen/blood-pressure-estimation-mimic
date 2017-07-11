function [ output_args ] = ple_filter( signal )
%PLE_FILTER 이 함수의 요약 설명 위치
%   자세한 설명 위치

output_args = - 1;

idx = length(signal);

for i = 1:idx
    if max(signal) < 0.8 && min(signal) > -0.8
        if signal(1) > -0.8 && signal(1) < 0
            if signal(20) > -0.5 && signal(20) < 0.75
                if signal(30) > -0.05 && signal(30) <  0.75
                    if signal(50) < 0.4 && signal(50) > -0.4
                        if signal(60) > -0.6 && signal(60) < 0.3
                            if signal(70) < 0.15 && signal(70) > -0.65
                                output_args = signal;
                                for j = 40:5:length(signal)
                                    if diff ([signal(j-5), signal(j)]) > 0
                                        output_args = -1;
                                        break;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

end