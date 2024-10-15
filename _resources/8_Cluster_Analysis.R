library(here)
library(ggplot2)
library(ggpubr)
library(rstatix)


here::i_am('8_Cluster_Analysis.R')

#Prepare df 
df <- read.csv("CSV/data.csv")

colnames(df)[4] <- "count"

df$cluster <- factor(df$cluster)
df$conditions <- factor(df$conditions)


# Stacked bar plot 
p <- ggbarplot(
  df, x = "cluster", y = "count", add = "mean_se",
  fill = "conditions", group = "donor" )+
  scale_fill_manual(values=c("white","darkmagenta","gray"),
                    labels = c("c1", "c2", "c3")) +
  labs(fill = "conditions")+
  guides(fill = guide_legend(title = NULL))+
  theme_classic()+
  theme(legend.position = 'top')

# Statistics
s <- df %>%
  group_by(cluster) %>%
  t_test(count ~ conditions) %>%
  adjust_pvalue() %>%
  add_significance()
s

# Filter for clusters
s2 <- s %>% filter(cluster %in% c(7, 8))

# Add p-values to the bar plot 
p<- p + stat_pvalue_manual(
  s2, x = "cluster", y.position = c(3250,3000,2750,4000,3750, 3500),
  label = "p.adj.signif"
)

#save png 
ggsave('png/cluster.png',p,width=5,height=3,dpi=300)

