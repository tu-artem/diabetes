
library(shiny)
library(DT)
library(glue)
library(shinydashboard)


source("utils.R")

ui <- dashboardPage(

   
  dashboardHeader(title="Hospital Readmission Control"),
   
   
  dashboardSidebar(

        numericInput("age",
                     "Enter age",
                      42),
        
        selectInput("gender", "Select gender", 
                    choices = list("Male" = "Male", 
                                   "Female" = "Female",
                                   "Other" = "Unknown/Invalid"), 
                    selected = 1),
        selectInput("race", "Select race", 
                    choices = list("Caucasian" = "Caucasian",
                                   "AfricanAmerican" = "AfricanAmerican",
                                   "Hispanic" = "Hispanic",
                                   "Asian" = "Asian",    
                                   "Other" = "Other"), 
                    selected = 1),
        
      # Input: Select a file ----
        fileInput("file1", "Upload medical details",
                  multiple = FALSE,
                  accept = c("text/csv",
                             "text/comma-separated-values,text/plain",
                             ".csv")),
      # Diagnoses
      selectizeInput(
        'diag', 
        label="Diagnoses",
        choices = diag_codes,
        multiple = TRUE,
        options = list(maxItems = 3, 
                       placeholder = 'Enter up to 3 Diagnoses')
      ),
      actionButton("predict", "Verify")
      ),
      
      
      dashboardBody(
        fluidRow(
          box(dataTableOutput("contents"),  collapsible = TRUE, width=12, height=180,
            title = "Data", status = "primary", solidHeader = TRUE)),
  
        fluidRow(
        column(width = 5, box(width=NULL, 
                              height=300,
                              h3(textOutput(("text_predictions"))))),

        column(width = 7, box(width = NULL, plotOutput("chart"), collapsible = TRUE,
            title = "Probabilities", status = "primary", solidHeader = TRUE, collapsed=TRUE))
      ))
   )



server <- function(input, output, session) {
  
  
  verify <- eventReactive(input$predict, {
    
    df <- tibble(race=input$race, 
                 gender=input$gender,
                 age=age_range(input$age))
    
    df2 <- read_csv(input$file1$datapath)
    
    df3 <- bind_cols(df, df2)
    
    predictions <- get_prediction(df3)
    predictions <<- predictions
    predictions
  })
  
  
   output$contents <- renderDataTable({
      req(input$file1)
     
     tryCatch(
       {
         df <- read.csv(input$file1$datapath)
       },
       error = function(e) {
         # return a safeError if a parsing error occurs
         stop(safeError(e))
       }
     )
     
    return(DT::datatable(df, options = list(scrollX = TRUE, dom = 't')))
   })
   
   
   output$text_predictions <- renderPrint({
   predictions <- verify()
     glue("Patient has {format(predictions[2] * 100, digits=3)}% chance of being readmitted")
   })
   
   output$chart <- renderPlot({
     predictions <- verify()
     data <- tibble(readmitted=c("NO", "YES"), probability=predictions)
     ggplot(data) + 
       geom_bar(aes(x=readmitted,y=probability), stat="identity") + 
       theme_minimal()
   })
}

# Run the application 
shinyApp(ui = ui, server = server)

