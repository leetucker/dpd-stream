* dpdsave: write e() of the last xtabond2 estimation (and, with uid = 1,
* r() of underid, kp sw) to the open file handle fh as one JSON line.
capture program drop dpdsave
program define dpdsave
    args name uid
    local q = char(34)
    tempname b V d
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colnames `b'
    local k = colsof(`b')
    file write fh "{" (`"`q'name`q': `q'`name'`q', `q'names`q': ["')
    forvalues i = 1/`k' {
        local nm : word `i' of `names'
        file write fh (`"`q'`nm'`q'"') (cond(`i' < `k', ", ", ""))
    }
    file write fh (`"], `q'b`q': ["')
    forvalues i = 1/`k' {
        file write fh %21.0g (`b'[1, `i']) (cond(`i' < `k', ", ", ""))
    }
    file write fh (`"], `q'V`q': ["')
    forvalues i = 1/`k' {
        forvalues j = 1/`k' {
            file write fh %21.0g (`V'[`i', `j']) (cond(`i' < `k' | `j' < `k', ", ", ""))
        }
    }
    file write fh "]"
    foreach s in N N_g j sargan sar_df hansen hansen_df ar1 ar2 sig2 chi2 F df_m df_r g_min g_max zrank zrank_nl {
        if e(`s') < . {
            file write fh (`", `q'`s'`q': "') %21.0g (e(`s'))
        }
    }
    if "`e(cmd)'" == "xtdpdgmm" {
        * what xtdpdgmm keeps of its estimation: the objective, sigma^2, the type of
        * variance correction and the weighting matrix (to compare with ours)
        foreach s in f chi2_J chi2_J_u sigma2e steps {
            if e(`s') < . {
                file write fh (`", `q'`s'`q': "') %21.0g (e(`s'))
            }
        }
        file write fh (`", `q'vcecor`q': `q'`e(vcecor)'`q', `q'vce`q': `q'`e(vce)'`q'"')
        file write fh (`", `q'estimator`q': `q'`e(estimator)'`q'"')
        capture confirm matrix e(W)
        if !_rc {
            matrix `d' = e(W)
            local rr = rowsof(`d')
            local cc = colsof(`d')
            file write fh (`", `q'W_dim`q': ["') %9.0f (`rr') ", " %9.0f (`cc') "]"
            file write fh (`", `q'W`q': ["')
            forvalues i = 1/`rr' {
                forvalues j = 1/`cc' {
                    file write fh %21.0g (`d'[`i', `j']) (cond(`i' < `rr' | `j' < `cc', ", ", ""))
                }
            }
            file write fh "]"
        }
        capture confirm matrix e(V_modelbased)
        if !_rc {
            matrix `d' = e(V_modelbased)
            local rr = rowsof(`d')
            file write fh (`", `q'V_modelbased`q': ["')
            forvalues i = 1/`rr' {
                forvalues j = 1/`rr' {
                    file write fh %21.0g (`d'[`i', `j']) (cond(`i' < `rr' | `j' < `rr', ", ", ""))
                }
            }
            file write fh "]"
        }
    }
    capture confirm matrix e(diffsargan)
    if !_rc {
        matrix `d' = e(diffsargan)
        local g = colsof(`d')
        file write fh (`", `q'diffsargan`q': ["')
        forvalues c = 1/`g' {
            file write fh "[" %21.0g (`d'[1, `c']) ", " %21.0g (`d'[2, `c']) ", " ///
                %21.0g (`d'[3, `c']) "]" (cond(`c' < `g', ", ", ""))
        }
        file write fh "]"
    }
    if "`uid'" == "1" {
        capture noisily underid, kp sw
        if !_rc {
            file write fh (`", `q'kp_lm`q': "') %21.0g (r(j_uid)) ///
                (`", `q'kp_df`q': "') %21.0g (r(df_uid))
            capture confirm matrix r(sw_uid)
            if !_rc {
                matrix `d' = r(sw_uid)
                local rr = rowsof(`d')
                local cc = colsof(`d')
                file write fh (`", `q'sw`q': ["')
                forvalues i = 1/`rr' {
                    file write fh "["
                    forvalues j = 1/`cc' {
                        file write fh %21.0g (`d'[`i', `j']) (cond(`j' < `cc', ", ", ""))
                    }
                    file write fh "]" (cond(`i' < `rr', ", ", ""))
                }
                file write fh "]"
            }
        }
    }
    file write fh "}" _n
end
