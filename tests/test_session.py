from session import Session




""" Create a test session instance, do simple and complex extraction,
     output extraction level """
if __name__ == '__main__':
        root = '/home/tomhagley/Documents/SLEAPProject'
        session = '2022-12-14_ADU-M-0003'
        session1 = Session(root, session)
        session1.data_filtering('all')
        session1.extract_basic_data()
        session1.more_complex_extraction()
        session1.get_extraction_level()
        session1.change_of_mind()