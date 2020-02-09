classdef singleobjectracker
    %SINGLEOBJECTRACKER is a class containing functions to track a single
    %object in clutter. 
    %Model structures need to be called:
    %sensormodel: a structure specifies the sensor parameters
    %           P_D: object detection probability --- scalar
    %           lambda_c: average number of clutter measurements per time
    %           scan, Poisson distributed --- scalar 
    %           pdf_c: clutter (Poisson) density --- scalar
    %           intensity_c: clutter (Poisson) intensity --- scalar
    %motionmodel: a structure specifies the motion model parameters
    %           d: object state dimension --- scalar
    %           F: function handle return transition/Jacobian matrix
    %           f: function handle return predicted object state
    %           Q: motion noise covariance matrix
    %measmodel: a structure specifies the measurement model parameters
    %           d: measurement dimension --- scalar
    %           H: function handle return transition/Jacobian matrix
    %           h: function handle return the observation of the object
    %           state 
    %           R: measurement noise covariance matrix
    
    properties
        gating      %specify gating parameter
        reduction   %specify hypothesis reduction parameter
        density     %density class handle
    end
    
    methods
        
        function obj = initialize(obj,density_class_handle,P_G,m_d,w_min,merging_threshold,M)
            %INITIATOR initializes singleobjectracker class
            %INPUT: density_class_handle: density class handle
            %       P_G: gating size in decimal --- scalar
            %            in this case, P_G = Pr[ d^2 <= G ] = significance
            %            level. The larger P_G, the G. P_G here is the
            %            significance level. Common values is 0.99
            %       m_d: measurement dimension --- scalar
            %       wmin: allowed minimum hypothesis weight --- scalar
            %       merging_threshold: merging threshold --- scalar
            %       M: allowed maximum number of hypotheses --- scalar
            %OUTPUT:  obj.density: density class handle
            %         obj.gating.P_G: gating size in decimal --- scalar
            %         obj.gating.size: gating size --- scalar
            %         obj.reduction.w_min: allowed minimum hypothesis
            %         weight in logarithmic scale --- scalar 
            %         obj.reduction.merging_threshold: merging threshold
            %         --- scalar 
            %         obj.reduction.M: allowed maximum number of hypotheses
            %         --- scalar 
            
            obj.density = density_class_handle;
            obj.gating.P_G = P_G;
            obj.gating.size = chi2inv(obj.gating.P_G,m_d);
            obj.reduction.w_min = log(w_min);
            obj.reduction.merging_threshold = merging_threshold;
            obj.reduction.M = M;
        end
        
        function [estimates_x, estimates_P] = nearestNeighbourFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %NEARESTNEIGHBOURFILTER tracks a single object using nearest
            %neighbor association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of  
            %            size (measurement dimension) x (number of
            %            measurements at corresponding time step) 
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1   
            n = size(Z, 1);
            prior_state = state;
            P_D = sensormodel.P_D;
            lambda_c = sensormodel.intensity_c; 

            estimates = cell(n, 1);

            for step = 1 : n
                z = Z{step};
                % apply gating
                [z_ingate, meas_in_gate] = obj.density.ellipsoidalGating(prior_state, z, measmodel, obj.gating.size);
                num_meas = size(z_ingate, 2);
                if num_meas > 0
                    predictedLikelihood_log =  obj.density.predictedLikelihood(prior_state,z,measmodel);
                    [max_Likelihood_log, idx] = max(predictedLikelihood_log);
                    theta_max = P_D * exp(max_Likelihood_log) / lambda_c;
                    if theta_max > (1 - P_D) % if nearest measurement has highest probability
                        z_nn = z(:, idx);
                        estimates{step} = obj.density.update(prior_state, z_nn, measmodel);
                    else
                        estimates{step} = prior_state;
                    end
                else
                    estimates{step} = prior_state;
                end
                prior_state = obj.density.predict(estimates{step}, motionmodel);
            end
            estimates_x = cell(n, 1);
            estimates_P = cell(n, 1);
            for step = 1 : n
                estimates_x{step} = estimates{step}.x;
                estimates_P{step} = estimates{step}.P;
            end
        end
        
        
        function [estimates_x, estimates_P] = probDataAssocFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %PROBDATAASSOCFILTER tracks a single object using probalistic
            %data association 
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1  
            n = size(Z, 1);
            prior_state = state;
            P_D = sensormodel.P_D;
            lambda_c = sensormodel.intensity_c; 

            estimates = cell(n, 1);

            for step = 1 : n
                z = Z{step};
                % apply gating
                [z_ingate, meas_in_gate] = obj.density.ellipsoidalGating(prior_state, z, measmodel, obj.gating.size);
                num_meas = size(z_ingate, 2);
                num_hypothesis = num_meas + 1;
                hypotheses = struct([]);
                hypothesesWeight = [];
                for i = 1 : num_hypothesis
                    if i ~= num_hypothesis
                        predictedLikelihood_log =  obj.density.predictedLikelihood(prior_state, z_ingate(:, i), measmodel);
                        hypothesis = obj.density.update(prior_state, z_ingate(:, i), measmodel);
                        hypotheses = [hypotheses; hypothesis];
                        hypothesesWeight = [hypothesesWeight; log(P_D) + predictedLikelihood_log - log(lambda_c)];
                    else
                        hypothesis = prior_state;
                        hypotheses = [hypotheses; hypothesis];
                        hypothesesWeight = [hypothesesWeight; log(1 - P_D)];
                    end
                end
                
                [hypothesesWeight, ~] = normalizeLogWeights(hypothesesWeight);

                
                [hypothesesWeight, hypotheses] = hypothesisReduction.prune(hypothesesWeight, hypotheses, obj.reduction.w_min);
                [hypothesesWeight, ~] = normalizeLogWeights(hypothesesWeight);
                
                estimates{step} = obj.density.momentMatching(hypothesesWeight, hypotheses);
               
                % predict
                prior_state = obj.density.predict(estimates{step}, motionmodel);
            end
            estimates_x = cell(n, 1);
            estimates_P = cell(n, 1);
            for step = 1 : n
                estimates_x{step} = estimates{step}.x;
                estimates_P{step} = estimates{step}.P;
            end            
 
        end
        
        function [estimates_x, estimates_P] = GaussianSumFilter(obj, state, Z, sensormodel, motionmodel, measmodel)
            %GAUSSIANSUMFILTER tracks a single object using Gaussian sum
            %filtering
            %INPUT: state: a structure with two fields:
            %                x: object initial state mean --- (object state
            %                dimension) x 1 vector 
            %                P: object initial state covariance --- (object
            %                state dimension) x (object state dimension)
            %                matrix  
            %       Z: cell array of size (total tracking time, 1), each
            %       cell stores measurements of size (measurement
            %       dimension) x (number of measurements at corresponding
            %       time step)  
            %OUTPUT:estimates: cell array of size (total tracking time, 1),
            %       each cell stores estimated object state of size (object
            %       state dimension) x 1
            n = size(Z, 1);
            prior_hypotheses = [state];
            prior_weights = [0];
            P_D = sensormodel.P_D;
            lambda_c = sensormodel.intensity_c; 
            
            estimates = cell(n, 1);
            for step = 1 : n
                z = Z{step};

                num_priorHypotheses = length(prior_hypotheses);
                newHypotheses = struct([]);
                newHypothesesWeight = [];
                for ii = 1 : num_priorHypotheses
                    prior_state = prior_hypotheses(ii);
                    [z_ingate, meas_in_gate] = obj.density.ellipsoidalGating(prior_state, z, measmodel, obj.gating.size);
                    num_meas = size(z_ingate, 2);
                    num_subHypotheses = num_meas + 1;
                    sub_newHypotheses = struct([]);
                    sub_newHypothesesWeight = [];
                    % create hypothesis for each prior hypothesis
                    for jj = 1 : num_subHypotheses
                        if jj ~= num_subHypotheses
                            predictedLikelihood_log =  obj.density.predictedLikelihood(prior_state, z_ingate(:, jj), measmodel);
                            hypothesis = obj.density.update(prior_state, z_ingate(:, jj), measmodel);
                            sub_newHypotheses = [sub_newHypotheses; hypothesis];
                            sub_newHypothesesWeight = [sub_newHypothesesWeight; prior_weights(ii)+ log(P_D) + predictedLikelihood_log - log(lambda_c)];
                        else
                            hypothesis = prior_state;
                            sub_newHypotheses = [sub_newHypotheses; hypothesis];
                            sub_newHypothesesWeight = [sub_newHypothesesWeight; prior_weights(ii) + log(1 - P_D)];
                        end
                    end
                    newHypotheses = [newHypotheses; sub_newHypotheses];
                    newHypothesesWeight = [newHypothesesWeight; sub_newHypothesesWeight];
                end
            
                [newHypothesesWeight, ~] = normalizeLogWeights(newHypothesesWeight);
                % prune
                [newHypothesesWeight, newHypotheses] = hypothesisReduction.prune(newHypothesesWeight, newHypotheses, obj.reduction.w_min);
                % merge
                [newHypothesesWeight, newHypotheses] = hypothesisReduction.merge(newHypothesesWeight, newHypotheses, obj.reduction.merging_threshold, obj.density);
                % cap 
                [newHypothesesWeight, newHypotheses] = hypothesisReduction.cap(newHypothesesWeight, newHypotheses, obj.reduction.M);

                [M, idx] = max(newHypothesesWeight);
                estimates{step} = newHypotheses(idx);

                prior_hypotheses = struct([]);
                prior_weights = [];
                % predict
                for kk = 1 : size(newHypotheses, 1)
                    prior_state = obj.density.predict(newHypotheses(kk), motionmodel);
                    prior_hypotheses = [prior_hypotheses; prior_state];
                    prior_weights = [prior_weights; newHypothesesWeight];
                end
            end

            estimates_x = cell(n, 1);
            estimates_P = cell(n, 1);
            for step = 1 : n
                estimates_x{step} = estimates{step}.x;
                estimates_P{step} = estimates{step}.P;
            end      

        end
    end
end

