function [] = plot_data_pdfs(P,Q)

[fP, xP] = hist(P, 100);
plot(xP, fP/trapz(xP,fP), 'b', 'LineWidth', 1);

[fQ, xQ] = hist(Q, 100);
plot(xQ, fQ/trapz(xQ,fQ), 'r--', 'LineWidth', 1);
