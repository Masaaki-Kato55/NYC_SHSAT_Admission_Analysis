## Finding:
- One surprising finding is how school spending negatively impacts SPS acceptance. This will be further explained in the Methods/Results section.

### Effect of School Size:
- Correlation between the number of applications and HSPHS acceptances (r=0.81)
  ![Correlation between Number of applications and HSPHS acceptances](../old_code/images/correlation_applicationAdmission.png)
- Correlation between class size and HSPHS acceptances (r=0.36)/achievement scores (r=0.21)
  ![Correlation between class size and HSPHS acceptances](../old_code/images/correlations_multiple.png)

### Effect of School Spending:
- As shown in the image above, school spending has a negative impact on both achievement scores (r=-0.15) and HSPHS acceptances (r=-0.34).
- In addition, I conducted an independent t-test to test whether or not school spending impacts HSPHS acceptances
    - Compared the means of HSPHS acceptances between poor and rich schools. The groups were artifically divided using the median of the `spending per student` (column E).
    - Based on the t-test results, I concluded that scholl spending negatively impacts (t=-6.9,p<0.01) HSPHS acceptances
- Although this may seem counterintuitive, when you consider the negative correlation between class size and school spending (r=-0.46), it illustrates a clearer picture.

![corr](../old_code/images/correlation_multiple2png.png)

- Larger schools have a disproportional per student spending when compared to that of smaller schools, and because these large schools have higher acceptances, it makes it seem as if more spending leads to lower acceptances. So, if the per student spending was proportional to the class size across all schools, the school spending may positively impact HSPHS acceptances

